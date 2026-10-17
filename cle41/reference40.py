"""Les références des graines 0 à 23 : les mesures des courses publiées de la clé 40 (perceptron, 32×32) pour
SYS, INDEP, SEUL, GRAND et COURT : python3 reference40.py → reference40.json.
Lit ../cle40/resultats/tranche_P*.json et tranche_R*.json (les fichiers publiés de la clé 40)."""
import glob, json, os
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
    D = {}
    for f in sorted(glob.glob(os.path.join(ICI, "..", "cle40", "resultats", "tranche_[PR]*.json"))):
        D.update(json.load(open(f)))
    R = {k: mesures(v) for k, v in D.items() if k.startswith("mlp_") and k.split("_")[1] in
         ("SYS", "INDEP", "SEUL", "GRAND", "COURT")}
    assert len(R) == 5 * 24, len(R)
    json.dump(R, open(os.path.join(ICI, "reference40.json"), "w"), indent=0, sort_keys=True)
    print(len(R), "courses de référence")
