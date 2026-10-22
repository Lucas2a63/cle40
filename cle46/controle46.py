"""Contrôle de l'environnement avant de lancer une tranche : python3 controle46.py

Rejoue trois débuts de course du réseau convolutif en 32×32, sur la graine 200 (hors plan), et les compare, valeur
par valeur, à ce qu'ils donnaient dans le bac à sable où le plan a été écrit (controle46_attendu.json) :
  1. INDEPC, 2 générations, avec les mesures finales et le choix final (les sept règles, leurs fronts, leur
     diversité) ;
  2. SEUL10, 2 générations, avec les mesures finales (le réseau seul, au taux de la clé 40) ;
  3. GRAND, 2 générations, avec les mesures finales (le grand réseau, au taux de la clé 40).
Affiche « bon » pour chacun si l'écart maximal est nul, sinon l'écart. Quelques minutes.
(python3 controle46.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle46 as E

CHOIX40 = json.load(open(os.path.join(ICI, "balayage40_choix.json")))
CAS = {
    "indepc": dict(bras="INDEPC", arch="cnn", seed=200, gens=2, finales=True),
    "seul10": dict(bras="SEUL10", arch="cnn", seed=200, gens=2, finales=True, lr0=E.LR0 * float(CHOIX40["SEUL10_cnn"])),
    "grand": dict(bras="GRAND", arch="cnn", seed=200, gens=2, finales=True, lr0=E.LR0 * float(CHOIX40["GRAND_cnn"])),
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
    c = r.get("fin_choix")
    if c:
        out += [float(x) for x in c["indiv"]]
        for k in sorted(c["reseaux"]):
            out += [float(x) for x in c["reseaux"][k]]
        out += [float(x) for x in c["glouton"]["chemin"]] + [float(c["glouton"]["etape"])]
        out += [float(x) for x in c["elagage"]["chemin"]]
        for nom in E.REGLES:
            g = c["regles"][nom]
            out += [float(x) for x in g["m"]] + [float(g["nb"]), float(g["crit"])]
            out += [float(x) for x in g["front"]["Rf"] + g["front"]["Ao"] + g["ancre"]["note"]]
            out += [float(g["diversite"][k]) for k in sorted(g["diversite"])]
    return out


if __name__ == "__main__":
    chemin = os.path.join(ICI, "controle46_attendu.json")
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
            print(f"{k} : nombre de valeurs différent ({len(b)} contre {len(a)} attendues)")
            ok = False
            continue
        ecart = max(abs(x - y) for x, y in zip(a, b))
        print(f"{k} : " + ("bon" if ecart == 0 else f"ÉCART {ecart:.3g}"), flush=True)
        ok = ok and ecart == 0
    sys.exit(0 if ok else 1)
