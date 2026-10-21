"""Contrôle de l'environnement avant de lancer une tranche : python3 controle44.py

Rejoue quatre débuts de course du perceptron en 32×32, sur la graine 200 (hors plan), et les compare, valeur par
valeur, à ce qu'ils donnaient dans le bac à sable où le plan a été écrit (controle44_attendu.json) :
  1. ENS, 2 générations, avec les mesures finales — la sélection pour l'ensemble, la diversité, les fronts de
     l'ensemble ;
  2. ENSR, 2 générations — la même sélection, qui ne copie que les réglages ;
  3. INDEP, 2 générations — la recherche aléatoire, avec la diversité à chaque passage du juge ;
  4. M32, 1 génération (taux LR0/8, décroissance 3e-2) — 32 recettes réglées en parallèle.
Affiche « bon » pour chacun si l'écart maximal est nul, sinon l'écart. Environ quatre minutes.
(python3 controle44.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle44 as E

CAS = {
    "ens": dict(bras="ENS", arch="mlp", seed=200, gens=2, finales=True),
    "ensr": dict(bras="ENSR", arch="mlp", seed=200, gens=2, finales=False),
    "indep": dict(bras="INDEP", arch="mlp", seed=200, gens=2, finales=False),
    "m32": dict(bras="M32", arch="mlp", seed=200, gens=1, finales=False, lr0=E.LR0 / 8, wd=3e-2),
}


def valeurs(r):
    out = []
    for d in r["tr"] + r["juge"]:
        out += [float(d[k]) for k in sorted(d) if isinstance(d[k], (int, float))]
    f = r.get("fin_livre")
    if f:
        out += [float(x) for x in f["Rf"] + f["Ao"] + f["ancre"]["note"]]
    e = r.get("fin_ens")
    if e:
        out += [float(e["V30_ens"])] + [float(x) for x in e["front"]["Rf"] + e["front"]["Ao"] + e["ancre"]["note"]]
        out += [float(e["diversite"][k]) for k in sorted(e["diversite"])]
    return out


if __name__ == "__main__":
    chemin = os.path.join(ICI, "controle44_attendu.json")
    calc = {k: valeurs(E.course(res=32, **kw)) for k, kw in CAS.items()}
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
