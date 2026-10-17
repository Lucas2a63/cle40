"""Les mesures des courses publiées de la clé 39 (16×16), pour les comparaisons d'échelle de la clé 40 :
python3 reference39.py ../cle39/resultats → reference39.json
Mêmes définitions que verdict39.py et verdict40.py (V_fin, V_vie, H, E_fin)."""
import glob, json, os, sys
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
TARDIF = [g for g in GRILLE if g >= 40]


def mesures(r):
    j = {d["g"]: d for d in r["juge"]}
    out = dict(V_fin=j[119]["V30_b"], V_vie=float(np.mean([j[g]["V30_b"] for g in GRILLE])),
               H=float(np.mean([j[g]["note_b"] - j[g]["V30_b"] for g in TARDIF])))
    if "V30_ens" in j[119]:
        out["E_fin"] = j[119]["V30_ens"]
    return out


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ICI, "..", "cle39", "resultats")
    D = {}
    for f in sorted(glob.glob(os.path.join(d, "tranche_*.json"))):
        D.update(json.load(open(f)))
    out = {k: mesures(v) for k, v in sorted(D.items())}
    json.dump(out, open(os.path.join(ICI, "reference39.json"), "w"), indent=0, sort_keys=True)
    print(len(out), "courses de la clé 39")
