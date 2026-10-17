"""Contrôle de l'environnement avant de lancer une tranche : python3 controle42.py

Rejoue quatre débuts de course du perceptron en 32×32, sur la graine 97 (hors plan), et les compare, valeur par
valeur, à ce qu'ils donnaient dans le bac à sable où le plan a été écrit (controle42_attendu.json) :
  1. SYSL, 2 générations — le système au génome élargi ;
  2. SYSR, 2 générations — le système au génome parti de la recette réglée ;
  3. M32, 1 génération (taux LR0/8, décroissance 3e-2) — 32 recettes réglées en parallèle ;
  4. COURT2, course entière avec ses mesures finales (taux LR0/8, décroissance 3e-2) — dont le front du juge et le
     front de l'ancre.
Affiche « bon » pour chacun si l'écart maximal est nul, sinon l'écart. Environ trois minutes.
(python3 controle42.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle42 as E

CAS = {
    "sysl": dict(bras="SYSL", arch="mlp", seed=97, gens=2, finales=False),
    "sysr": dict(bras="SYSR", arch="mlp", seed=97, gens=2, finales=False),
    "m32": dict(bras="M32", arch="mlp", seed=97, gens=1, finales=False, lr0=E.LR0 / 8, wd=3e-2),
    "court2": dict(bras="COURT2", arch="mlp", seed=97, finales=True, lr0=E.LR0 / 8, wd=3e-2),
}


def valeurs(r):
    out = []
    for d in r["tr"] + r["juge"]:
        out += [float(d[k]) for k in sorted(d) if isinstance(d[k], (int, float))]
    f = r.get("fin_livre")
    if f:
        out += [float(x) for x in f["Rf"] + f["Ao"] + f["ancre"]["note"]]
    return out


if __name__ == "__main__":
    chemin = os.path.join(ICI, "controle42_attendu.json")
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
