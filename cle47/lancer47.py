"""Exécute une tranche du plan de la clé 47, avec points de reprise : python3 lancer47.py --tranche K [--processus auto|N]

Chaque course passe par reprise47.course, qui écrit l'état de la course dans donnees/points/<course>.pkl à la fin de
chaque génération (et après chaque étape des mesures finales) et reprend de là si le fichier existe (donnees/ est
ignoré par git : les points ne sont jamais commités). Après un redémarrage du conteneur, relancer la même commande
reprend chaque course en cours là où elle s'était arrêtée, au lieu de la recommencer ; les courses finies (dans
resultats/tranche_K.json) ne sont pas refaites. Le point d'une course est effacé une fois son résultat écrit.
« duree » est le temps de calcul cumulé d'une course sur toutes ses reprises ; « tache » est la course du plan.
Les courses les plus longues partent les premières.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, json, multiprocessing as mp, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
ORDRE = {b: i for i, b in enumerate(("POPC", "K4P", "K32", "K8", "K2", "SOUPE", "TOILE", "VEILLEUR"))}
GO_PAR_PROCESSUS = 2.0


def une_course(t):
    import reprise47, plan47
    os.makedirs(os.path.join(ICI, "donnees", "points"), exist_ok=True)
    point = os.path.join(ICI, "donnees", "points", plan47.nom(t) + ".pkl")
    fduree = point[:-4] + ".duree"
    avant = float(open(fduree).read()) if os.path.exists(fduree) and os.path.exists(point) else 0.0
    t0 = time.time()
    import echelle47 as E

    class Compteur(E.Point):                             # le temps cumulé, écrit à chaque sauvegarde du point
        def sauver(self, **maj):
            super().sauver(**maj)
            with open(fduree + ".tmp", "w") as f:
                f.write(repr(avant + time.time() - t0))
            os.replace(fduree + ".tmp", fduree)

    ancien_point, ancien_ecrire = E.Point, reprise47._ecrire

    def ecrire(chemin, obj):                             # POPC : les points de reprise47 (course_pop_r)
        ancien_ecrire(chemin, obj)
        with open(fduree + ".tmp", "w") as f:
            f.write(repr(avant + time.time() - t0))
        os.replace(fduree + ".tmp", fduree)

    E.Point, reprise47._ecrire = Compteur, ecrire
    try:
        r = reprise47.course(t["bras"], t["arch"], int(t["seed"]), finales=True, res=int(t["res"]), point=point)
    finally:
        E.Point, reprise47._ecrire = ancien_point, ancien_ecrire
    r["duree"] = avant + time.time() - t0
    r["tache"] = dict(t)
    return plan47.nom(t), r


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
    import plan47
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", required=True)
    ap.add_argument("--processus", default="auto")
    ap.add_argument("--plan", default=os.path.join(ICI, "plan47.json"))
    a = ap.parse_args()
    taches = json.load(open(a.plan))["tranches"][a.tranche]
    os.makedirs(os.path.join(ICI, "resultats"), exist_ok=True)
    chemin = os.path.join(ICI, "resultats", f"tranche_{a.tranche}.json")
    out = json.load(open(chemin)) if os.path.exists(chemin) else {}
    reste = [t for t in taches if plan47.nom(t) not in out]
    reste.sort(key=lambda t: ORDRE[t["bras"]])
    coeurs = os.cpu_count() or 1
    mem = memoire_disponible_go()
    if a.processus == "auto":
        n = coeurs if mem is None else max(1, min(coeurs, int(mem // GO_PAR_PROCESSUS)))
    else:
        n = int(a.processus)
    n = max(1, min(n, len(reste) or 1))
    print(f"machine : {coeurs} cœurs, {mem if mem is None else round(mem, 1)} Go disponibles, numpy {np.__version__}",
          flush=True)
    print(f"tranche {a.tranche} : {len(taches)} courses, {len(taches) - len(reste)} déjà faites, "
          f"{len(reste)} à faire, sur {n} processus", flush=True)
    t0 = time.time()
    if reste:
        with mp.get_context("spawn").Pool(n, maxtasksperchild=1) as pool:
            for k, r in pool.imap_unordered(une_course, reste):
                out[k] = r
                ecrire(chemin, out)
                for ext in (".pkl", ".duree"):
                    p = os.path.join(ICI, "donnees", "points", k + ext)
                    if os.path.exists(p):
                        os.remove(p)
                fait = sum(plan47.nom(t) in out for t in taches)
                print(f"  {k:<26} fait en {r['duree'] / 60:6.1f} min   [{fait}/{len(taches)}]   "
                      f"écoulé {(time.time() - t0) / 60:6.1f} min", flush=True)
    fait = sum(plan47.nom(t) in out for t in taches)
    if fait == len(taches):
        print(f"TRANCHE {a.tranche} TERMINÉE — {fait} courses dans resultats/tranche_{a.tranche}.json", flush=True)
    else:
        print(f"tranche {a.tranche} incomplète : {fait}/{len(taches)}", flush=True)
        sys.exit(1)
