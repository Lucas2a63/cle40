"""Exécute une tranche du plan de la clé 46, avec points de reprise : python3 lancer46r.py --tranche K [--processus auto|N]

Le même lanceur que lancer46.py (mêmes courses, mêmes réglages, même fichier de résultats, mêmes champs « duree » et
« tache »), à une différence près : chaque course passe par reprise46.course, qui écrit l'état de la course dans
donnees/points/<course>.pkl à la fin de chaque génération et reprend de là si le fichier existe (donnees/ est ignoré
par git : les points ne sont jamais commités). Après un redémarrage du conteneur, relancer la même commande reprend
chaque course en cours à la génération où elle s'était arrêtée, au lieu de la recommencer. Le point d'une course est
effacé une fois son résultat écrit. « duree » est le temps de calcul cumulé sur toutes les reprises d'une course.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, hashlib, json, multiprocessing as mp, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
POP = ("INDEPC", "ENS", "ENSR", "INDEP", "SYS", "INDEP0", "M32")
DUREE = {b: 0 for b in POP} | {"GRAND": 1, "GRAND2": 1, "SEUL10": 2, "SEUL": 2, "SEUL2": 2, "COURT": 3, "COURT2": 3}


def reglages(t, choix40, choix41):
    """(lr0, wd) d'une tâche ; pour les bras de population, LR0 et WD0 (leurs réglages viennent du génome)."""
    import echelle46 as E
    wd = lambda w: E.WD0 if w == "WD0" else float(w)
    b = t["bras"]
    if b in ("SEUL", "SEUL10", "GRAND", "COURT"):
        return E.LR0 * float(choix40[f"{b}_{t['arch']}"]), E.WD0
    if b in ("SEUL2", "GRAND2", "COURT2", "M32"):
        return E.LR0 * float(choix41[b]["mult"]), wd(choix41[b]["wd"])
    return E.LR0, E.WD0


def une_course(args):
    t, lr0, wd = args
    import reprise46, plan46
    os.makedirs(os.path.join(ICI, "donnees", "points"), exist_ok=True)
    point = os.path.join(ICI, "donnees", "points", plan46.nom(t) + ".pkl")
    fduree = point[:-4] + ".duree"
    avant = float(open(fduree).read()) if os.path.exists(fduree) and os.path.exists(point) else 0.0
    t0 = time.time()

    def compter():                                       # le temps cumulé, à chaque génération (via le point)
        with open(fduree + ".tmp", "w") as f:
            f.write(repr(avant + time.time() - t0))
        os.replace(fduree + ".tmp", fduree)

    import echelle46 as E
    ancien = reprise46._ecrire

    def ecrire(chemin, obj):
        ancien(chemin, obj)
        compter()

    reprise46._ecrire = ecrire
    r = reprise46.course(t["bras"], t["arch"], int(t["seed"]), finales=True, lr0=lr0, res=int(t["res"]), wd=wd,
                         point=point)
    r["duree"] = avant + time.time() - t0
    r["tache"] = dict(t, lr0=lr0, wd_valeur=wd)
    return plan46.nom(t), r


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
    import plan46
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", required=True)
    ap.add_argument("--processus", default="auto")
    ap.add_argument("--plan", default=os.path.join(ICI, "plan46.json"))
    a = ap.parse_args()
    taches = json.load(open(a.plan))["tranches"][a.tranche]
    choix40 = charger_choix("balayage40_choix.json", any(t["bras"] in ("SEUL", "SEUL10", "GRAND", "COURT") for t in taches))
    choix41 = charger_choix("balayage41_choix.json", any(t["bras"] in ("SEUL2", "GRAND2", "COURT2", "M32") for t in taches))
    os.makedirs(os.path.join(ICI, "resultats"), exist_ok=True)
    chemin = os.path.join(ICI, "resultats", f"tranche_{a.tranche}.json")
    out = json.load(open(chemin)) if os.path.exists(chemin) else {}
    reste = [t for t in taches if plan46.nom(t) not in out]
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
                for ext in (".pkl", ".duree"):
                    p = os.path.join(ICI, "donnees", "points", k + ext)
                    if os.path.exists(p):
                        os.remove(p)
                fait = sum(plan46.nom(t) in out for t in taches)
                print(f"  {k:<26} fait en {r['duree'] / 60:6.1f} min   [{fait}/{len(taches)}]   "
                      f"écoulé {(time.time() - t0) / 60:6.1f} min", flush=True)
    fait = sum(plan46.nom(t) in out for t in taches)
    if fait == len(taches):
        print(f"TRANCHE {a.tranche} TERMINÉE — {fait} courses dans resultats/tranche_{a.tranche}.json", flush=True)
    else:
        print(f"tranche {a.tranche} incomplète : {fait}/{len(taches)}", flush=True)
        sys.exit(1)
