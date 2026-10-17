"""Contrôle de l'environnement avant de lancer une tranche : python3 controle40.py

Rejoue quatre débuts de course en 32×32 et les compare, valeur par valeur, à ce qu'ils donnaient dans le bac à
sable où le plan a été écrit (controle40_attendu.json) :
  1. SYS, convolutif, graine 0, 1 génération — la population convolutive (les courses les plus longues) ;
  2. COURT, convolutif, graine 99, 2 générations — le code d'un réseau seul ;
  3. SYS, perceptron, graine 0, 2 générations — la population du perceptron ;
  4. GRAND, perceptron, graine 99, 1 génération — le réseau élargi.
Affiche « bon » pour chacun si l'écart maximal est nul, sinon l'écart. Environ cinq minutes.
(python3 controle40.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle40 as E

CAS = {
    "sys_cnn": dict(bras="SYS", arch="cnn", seed=0, gens=1),
    "court_cnn": dict(bras="COURT", arch="cnn", seed=99, gens=2),
    "sys_mlp": dict(bras="SYS", arch="mlp", seed=0, gens=2),
    "grand_mlp": dict(bras="GRAND", arch="mlp", seed=99, gens=1),
}


def valeurs(r):
    out = []
    for d in r["tr"] + r["juge"]:
        out += [float(d[k]) for k in sorted(d) if isinstance(d[k], (int, float))]
    return out


if __name__ == "__main__":
    chemin = os.path.join(ICI, "controle40_attendu.json")
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
