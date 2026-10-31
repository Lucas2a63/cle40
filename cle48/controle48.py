"""Contrôle de l'environnement avant de lancer une tranche : python3 controle48.py

Rejoue quatre débuts de course du réseau convolutif en 32×32, sur la graine 200 (hors plan), avec toutes leurs mesures
finales, et les compare à ce qu'ils donnaient dans le bac à sable où le plan a été écrit
(controle48_attendu.json) — le résultat entier, écrit en JSON, caractère par caractère (au bit près, −0 et NaN
compris) :
  1. PHOTO2, deux générations : deux réseaux qui se partagent la moitié du calcul, leur ensemble, chacun, la moyenne
     de leurs poids, leur confiance, puis la distillation de leur ensemble dans un seul réseau nourri de l'autre moitié ;
  2. K4_S, deux générations : quatre réseaux qui se partagent le seizième du calcul ;
  3. TOILE, deux cycles d'une génération : le réseau au taux cyclique, ses instantanés, leur ensemble, la moyenne de
     leurs poids, celle de la seconde moitié, la confiance ;
  4. K4_S12 : K4_S sur douze générations — la bascule ×30 de la 10e génération comprise.
Affiche « bon » pour chacun, ou le premier endroit où le résultat diffère ; puis « CONTRÔLE : bon » si tout est
identique. De quinze à trente minutes.
(python3 controle48.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle48 as E
import reprise48 as R

CAS = {
    "PHOTO2": lambda: R.course("PHOTO2", "cnn", 200, gens=2, finales=True),
    "K4_S": lambda: R.course("K4_S", "cnn", 200, gens=2, finales=True),
    "TOILE": lambda: E.course_toile48("cnn", 200, 10, 2, 2, True),
    "K4_S12": lambda: R.course("K4_S", "cnn", 200, gens=12, finales=True),
}


def texte(r):
    return json.dumps(r, sort_keys=True)


def premier_ecart(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"au caractère {i} : …{a[max(0, i - 60):i + 40]}… contre …{b[max(0, i - 60):i + 40]}…"
    return f"longueurs {len(a)} et {len(b)}"


if __name__ == "__main__":
    E.resolution(32)
    chemin = os.path.join(ICI, "controle48_attendu.json")
    calc = {}
    for b, f in CAS.items():
        calc[b] = texte(f())
        print(f"{b} : calculé", flush=True)
    if "--ecrire" in sys.argv:
        json.dump(calc, open(chemin, "w"))
        print("écrit", chemin)
        sys.exit(0)
    att = json.load(open(chemin))
    ok = True
    for b in CAS:
        bon = att[b] == calc[b]
        print(f"{b} : " + ("bon" if bon else "ÉCART " + premier_ecart(att[b], calc[b])), flush=True)
        ok = ok and bon
    print("CONTRÔLE :", "bon" if ok else "ÉCART", flush=True)
    sys.exit(0 if ok else 1)
