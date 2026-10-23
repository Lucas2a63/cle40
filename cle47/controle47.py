"""Contrôle de l'environnement avant de lancer une tranche : python3 controle47.py

Rejoue trois débuts de course du réseau convolutif en 32×32, sur la graine 200 (hors plan), deux générations chacun,
avec toutes leurs mesures finales, et les compare à ce qu'ils donnaient dans le bac à sable où le plan a été écrit
(controle47_attendu.json) — le résultat entier, écrit en JSON, caractère par caractère (au bit près, −0 et NaN
compris) :
  1. K4P : la recette coupée en quatre réseaux, leur ensemble, chacun, la moyenne de leurs poids, puis la distillation
     de leur ensemble dans un seul réseau ;
  2. SOUPE : le parent, les quatre enfants, la soupe, la soupe gourmande, l'ensemble des enfants ;
  3. POPC : la population faite pour le convolutif, avec le choix final de la clé 45.
Affiche « bon » pour chacun, ou le premier endroit où le résultat diffère ; puis « CONTRÔLE : bon » si tout est
identique. De quinze à trente minutes.
(python3 controle47.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import reprise47 as R

CAS = ("K4P", "SOUPE", "POPC")


def texte(r):
    return json.dumps(r, sort_keys=True)


def premier_ecart(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"au caractère {i} : …{a[max(0, i - 60):i + 40]}… contre …{b[max(0, i - 60):i + 40]}…"
    return f"longueurs {len(a)} et {len(b)}"


if __name__ == "__main__":
    chemin = os.path.join(ICI, "controle47_attendu.json")
    calc = {}
    for b in CAS:
        calc[b] = texte(R.course(b, "cnn", 200, gens=2, finales=True))
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
