"""Des données de synthèse pour essayer verdict48.py et ana48.py avant les courses : python3 synthese48.py DOSSIER

Fait chaque bras du plan une fois, au petit modèle du calcul (2 pas par génération, un juge de 2 500 images, 4
générations, 8 pour TOILE8 ; graine 401, hors plan), puis en tire 24 copies, une par tranche du plan, dont les fronts
(du juge et de l'ancre), les ambiguïtés et l'échelle des logits sont bruités au hasard — rien qui ressemble à un
résultat : de quoi vérifier que les scripts lisent toutes les mesures et appliquent la règle. Écrit
DOSSIER/tranche_J*.json.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import copy, json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import numpy as np
import echelle48 as E
import reprise48 as R
import plan48


def bruiter(x, rng, cle=None):
    """Bruite les fronts (listes Rf, Ao, note), les ambiguïtés et l'échelle des logits, partout dans un résultat."""
    if isinstance(x, dict):
        return {k: (bruiter(v, rng, k)) for k, v in x.items()}
    if isinstance(x, list):
        if cle in ("Rf", "Ao", "note") and x and isinstance(x[0], float):
            d = rng.normal(0, 0.01)
            return [float(min(1.0, max(0.0, v + d + rng.normal(0, 0.002)))) for v in x]
        return [bruiter(v, rng, cle) for v in x]
    if cle in ("amb", "amb_chat") and isinstance(x, float):
        return float(abs(x + rng.normal(0, 0.01)))
    if cle == "ecart_logits" and isinstance(x, float):
        return float(abs(x * (1 + rng.normal(0, 0.05))))
    return x


if __name__ == "__main__":
    dossier = sys.argv[1]
    os.makedirs(dossier, exist_ok=True)
    E.PAS = 2
    Xtr, ytr, Xte, yte = E.donnees()
    E._D["brut"] = (Xtr, ytr, Xte[:2500], yte[:2500])
    E._D.pop("juge_propre", None)
    base = {}
    for b in plan48.BRAS_A + plan48.BRAS_B:
        base[b] = R.course(b, "cnn", 401, gens=8 if b == "TOILE8" else 4)
        print(b, "fait", flush=True)
    rng = np.random.default_rng(4848)
    for k, taches in plan48.tranches().items():
        out = {}
        for t in taches:
            r = bruiter(copy.deepcopy(base[t["bras"]]), rng)
            r["duree"] = float(rng.uniform(0.2, 3) * 3600)
            r["tache"] = dict(t)
            out[plan48.nom(t)] = r
        json.dump(out, open(os.path.join(dossier, f"tranche_{k}.json"), "w"))
    print("écrit", dossier)
