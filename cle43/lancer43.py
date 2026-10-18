"""Exécute une tranche du plan de la clé 43 : python3 lancer43.py --tranche K [--processus auto|N]

Chaque course tourne dans son propre processus, sur un seul fil de calcul. Les résultats s'écrivent au fur et
à mesure dans resultats/tranche_K.json (écriture atomique). Une course déjà présente dans ce fichier n'est
pas refaite : relancer la même commande reprend là où l'on s'était arrêté.
Le taux de départ (et la décroissance) d'une recette standard vient de balayage40_choix.json (GRAND, SEUL) ou de
balayage41_choix.json (COURT2, M32) ; si le fichier manque, la course n'est pas lancée.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, hashlib, json, multiprocessing as mp, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
POP = ("SYS", "INDEP", "INDEP0", "M32")
DUREE = {b: 0 for b in POP} | {"GRAND": 1, "GRAND2": 1, "SEUL": 2, "SEUL2": 2, "COURT": 3, "COURT2": 3}


def reglages(t, choix40, choix41):
    """(lr0, wd) d'une tâche ; pour les bras de population, LR0 et WD0 (leurs réglages viennent du génome)."""
    import echelle43 as E
    wd = lambda w: E.WD0 if w == "WD0" else float(w)
    b = t["bras"]
    if b in ("SEUL", "GRAND", "COURT"):
        return E.LR0 * float(choix40[f"{b}_mlp"]), E.WD0
    if b in ("SEUL2", "GRAND2", "COURT2", "M32"):
        return E.LR0 * float(choix41[b]["mult"]), wd(choix41[b]["wd"])
    return E.LR0, E.WD0


def une_course(args):
    t, lr0, wd = args
    import echelle43 as E, plan43
    t0 = time.time()
    r = E.course(t["bras"], t["arch"], int(t["seed"]), finales=True, lr0=lr0, res=int(t["res"]), wd=wd)
    r["duree"] = time.time() - t0
    r["tache"] = dict(t, lr0=lr0, wd_valeur=wd)
    return plan43.nom(t), r


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


def charger_choix(nom, besoin):
    if not besoin:
        return None
    cp = os.path.join(ICI, nom)
    if not os.path.exists(cp):
        sys.exit(f"{nom} manque.")
    print(f"{nom} : sha256 {hashlib.sha256(open(cp, 'rb').read()).hexdigest()}", flush=True)
    return json.load(open(cp))


if __name__ == "__main__":
    import numpy as np
    import plan43
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", required=True)
    ap.add_argument("--processus", default="auto")
    ap.add_argument("--plan", default=os.path.join(ICI, "plan43.json"))
    a = ap.parse_args()
    taches = json.load(open(a.plan))["tranches"][a.tranche]
    choix40 = charger_choix("balayage40_choix.json", any(t["bras"] in ("SEUL", "GRAND", "COURT") for t in taches))
    choix41 = charger_choix("balayage41_choix.json", any(t["bras"] in ("SEUL2", "GRAND2", "COURT2", "M32") for t in taches))
    os.makedirs(os.path.join(ICI, "resultats"), exist_ok=True)
    chemin = os.path.join(ICI, "resultats", f"tranche_{a.tranche}.json")
    out = json.load(open(chemin)) if os.path.exists(chemin) else {}
    reste = [t for t in taches if plan43.nom(t) not in out]
    reste.sort(key=lambda t: DUREE[t["bras"]])
    coeurs = os.cpu_count() or 1
    mem = memoire_disponible_go()
    if a.processus == "auto":
        n = coeurs if mem is None else max(1, min(coeurs, int(mem // 1.0)))
    else:
        n = int(a.processus)
    n = max(1, min(n, len(reste) or 1))
    print(f"machine : {coeurs} cœurs, {mem if mem is None else round(mem, 1)} Go disponibles, numpy {np.__version__}",
          flush=True)
    print(f"tranche {a.tranche} : {len(taches)} courses, {len(taches) - len(reste)} déjà faites, "
          f"{len(reste)} à faire, sur {n} processus", flush=True)
    t0 = time.time()
    if reste:
        args = [(t,) + reglages(t, choix40, choix41) for t in reste]
        with mp.get_context("spawn").Pool(n, maxtasksperchild=1) as pool:
            for k, r in pool.imap_unordered(une_course, args):
                out[k] = r
                ecrire(chemin, out)
                fait = sum(plan43.nom(t) in out for t in taches)
                print(f"  {k:<26} fait en {r['duree'] / 60:6.1f} min   [{fait}/{len(taches)}]   "
                      f"écoulé {(time.time() - t0) / 60:6.1f} min", flush=True)
    fait = sum(plan43.nom(t) in out for t in taches)
    if fait == len(taches):
        print(f"TRANCHE {a.tranche} TERMINÉE — {fait} courses dans resultats/tranche_{a.tranche}.json", flush=True)
    else:
        print(f"tranche {a.tranche} incomplète : {fait}/{len(taches)}", flush=True)
        sys.exit(1)
