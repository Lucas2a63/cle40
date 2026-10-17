"""Contrôle de l'environnement avant de lancer une tranche : python3 controle41.py

Rejoue quatre débuts de course du perceptron en 32×32 et les compare, valeur par valeur, à ce qu'ils donnaient
dans le bac à sable où le plan a été écrit (controle41_attendu.json) :
  1. POIDS, graine 0, 2 générations — la sélection qui ne copie que les poids ;
  2. REGLAGES, graine 1, 2 générations — la sélection qui ne copie que les réglages ;
  3. M32, graine 2, 1 génération (taux LR0/8, décroissance 1e-3) — 32 recettes standard en parallèle ;
  4. GRAND2, graine 99, 1 génération (taux LR0/16, décroissance 1e-3) — un réseau seul réglé sur deux axes.
Affiche « bon » pour chacun si l'écart maximal est nul, sinon l'écart. Environ trois minutes.
(python3 controle41.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle41 as E

CAS = {
    "poids": dict(bras="POIDS", arch="mlp", seed=0, gens=2),
    "reglages": dict(bras="REGLAGES", arch="mlp", seed=1, gens=2),
    "m32": dict(bras="M32", arch="mlp", seed=2, gens=1, lr0=E.LR0 / 8, wd=1e-3),
    "grand2": dict(bras="GRAND2", arch="mlp", seed=99, gens=1, lr0=E.LR0 / 16, wd=1e-3),
}


def valeurs(r):
    out = []
    for d in r["tr"] + r["juge"]:
        out += [float(d[k]) for k in sorted(d) if isinstance(d[k], (int, float))]
    return out


if __name__ == "__main__":
    chemin = os.path.join(ICI, "controle41_attendu.json")
    calc = {k: valeurs(E.course(finales=False, res=32, **kw)) for k, kw in CAS.items()}
    if "--ecrire" in sys.argv:
        json.dump(calc, open(chemin, "w"))
        print("écrit", chemin)
        sys.exit(0)
    att = json.load(open(chemin))
    ok = True
    for k in CAS:
        a, b = att[k], calc[k]
        if len(a) != len(b):
            print(f"{k} : nombre de valeurs différent ({len(b)} contre {len(a)} attendues)"); ok = False; continue
        ecart = max(abs(x - y) for x, y in zip(a, b))
        print(f"{k} : " + ("bon" if ecart == 0 else f"ÉCART {ecart:.3g}"), flush=True)
        ok = ok and ecart == 0
    sys.exit(0 if ok else 1)
