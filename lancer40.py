"""Exécute une tranche du plan de la clé 40 : python3 lancer40.py --tranche K [--processus auto|N]
(ou une liste de courses : python3 lancer40.py --taches fichier.json --sortie nom)

Chaque course tourne dans son propre processus, sur un seul fil de calcul. Les résultats s'écrivent au fur et
à mesure dans resultats/tranche_K.json (écriture atomique). Une course déjà présente dans ce fichier n'est
pas refaite : relancer la même commande reprend là où l'on s'était arrêté.

Le nombre de processus (auto) : autant que de cœurs, dans la limite de la mémoire disponible (une course de
population convolutive en 32×32 demande environ 1,7 Go (on en compte 2), les autres moins).
Le taux de départ d'un réseau seul vient de balayage40_choix.json ; s'il manque, la course n'est pas lancée.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, hashlib, json, multiprocessing as mp, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
MEMOIRE_GO = {("cnn", "pop"): 2.0, ("cnn", "seul"): 1.0, ("mlp", "pop"): 1.0, ("mlp", "seul"): 1.0}
DUREE = {"SYS": 0, "INDEP": 0, "SEUL": 1, "SEUL10": 1, "GRAND": 2, "COURT": 3}


def genre(t):
    return (t["arch"], "pop" if t["bras"] in ("SYS", "INDEP") else "seul")


def taux(t, choix):
    import echelle40 as E
    if t.get("balayage"):
        return E.LR0 * float(t["mult"])
    if t["bras"] in ("SYS", "INDEP"):
        return E.LR0
    return E.LR0 * float(choix[f"{t['bras']}_{t['arch']}"])


def une_course(args):
    t, lr0 = args
    import echelle40 as E, plan40
    t0 = time.time()
    r = E.course(t["bras"], t["arch"], int(t["seed"]), finales=not t.get("balayage"), lr0=lr0, res=int(t["res"]))
    if t.get("balayage"):
        notes = [d["note"] for d in r["tr"][-10:]]
        r["note_balayage"] = sum(notes) / len(notes)
    r["duree"] = time.time() - t0
    r["tache"] = dict(t, lr0=lr0)
    return plan40.nom(t), r


def ecrire(chemin, out):
    tmp = chemin + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, chemin)


def memoire_disponible_go():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


if __name__ == "__main__":
    import numpy as np
    import plan40
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tranche")
    g.add_argument("--taches")
    ap.add_argument("--sortie", default=None)
    ap.add_argument("--processus", default="auto")
    ap.add_argument("--plan", default=os.path.join(ICI, "plan40.json"))
    a = ap.parse_args()
    if a.tranche:
        taches = json.load(open(a.plan))["tranches"][a.tranche]
        sortie = a.tranche
    else:
        taches = json.load(open(a.taches))
        sortie = a.sortie or os.path.splitext(os.path.basename(a.taches))[0]
    choix = None
    if any(t["bras"] not in ("SYS", "INDEP") and not t.get("balayage") for t in taches):
        cp = os.path.join(ICI, "balayage40_choix.json")
        if not os.path.exists(cp):
            sys.exit("balayage40_choix.json manque : les taux des réseaux seuls ne sont pas encore fixés.")
        choix = json.load(open(cp))
        print(f"balayage40_choix.json : sha256 {hashlib.sha256(open(cp, 'rb').read()).hexdigest()}", flush=True)
    os.makedirs(os.path.join(ICI, "resultats"), exist_ok=True)
    chemin = os.path.join(ICI, "resultats", f"tranche_{sortie}.json")
    out = json.load(open(chemin)) if os.path.exists(chemin) else {}
    reste = [t for t in taches if plan40.nom(t) not in out]
    reste.sort(key=lambda t: (t["arch"] != "cnn", DUREE[t["bras"]], -float(t.get("mult") or 0)))
    coeurs = os.cpu_count() or 1
    mem = memoire_disponible_go()
    if a.processus == "auto":
        besoin = max([MEMOIRE_GO[genre(t)] for t in reste] or [1.0])
        n = coeurs if mem is None else max(1, min(coeurs, int(mem // besoin)))
    else:
        n = int(a.processus)
    n = max(1, min(n, len(reste) or 1))
    print(f"machine : {coeurs} cœurs, {mem if mem is None else round(mem, 1)} Go disponibles, numpy {np.__version__}",
          flush=True)
    print(f"tranche {sortie} : {len(taches)} courses, {len(taches) - len(reste)} déjà faites, "
          f"{len(reste)} à faire, sur {n} processus", flush=True)
    t0 = time.time()
    if reste:
        with mp.get_context("spawn").Pool(n, maxtasksperchild=1) as pool:
            for k, r in pool.imap_unordered(une_course, [(t, taux(t, choix)) for t in reste]):
                out[k] = r
                ecrire(chemin, out)
                fait = sum(plan40.nom(t) in out for t in taches)
                print(f"  {k:<24} fait en {r['duree'] / 60:6.1f} min   [{fait}/{len(taches)}]   "
                      f"écoulé {(time.time() - t0) / 60:6.1f} min", flush=True)
    fait = sum(plan40.nom(t) in out for t in taches)
    if fait == len(taches):
        print(f"TRANCHE {sortie} TERMINÉE — {fait} courses dans resultats/tranche_{sortie}.json", flush=True)
    else:
        print(f"tranche {sortie} incomplète : {fait}/{len(taches)}", flush=True)
        sys.exit(1)
