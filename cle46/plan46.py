"""Le plan de la clé 46 : python3 plan46.py écrit plan46.json. Tout est sur le réseau convolutif de la clé 40
(32×32×3 → 16×16×16 → 16×16×32 → 8×8×32 → 10), en 32×32, dans le monde des clés 40 à 45, sur 24 graines, 100 à 123
(les premières des clés 43 à 45, dont les courses du perceptron servent de référence, graine par graine).

CINQ BRAS, refaits sur chaque graine (120 courses ; tranches H1 à H12, deux graines et dix courses par tranche)
  INDEPC  la recherche aléatoire convolutive (avec curation, sans sélection ; ×30 dans la perte à partir de la
          10e génération), suivie du choix final de la clé 45 (sept règles, dont TOUS, l'ensemble des 32) ;
  SEUL10  le réseau convolutif seul, qui reçoit le calcul d'entraînement de la population, pesé à la 10e génération
          (la recette standard munie de la bascule) ;
  SEUL    le même, pesé dès le départ ;
  GRAND   le grand réseau convolutif (96 et 186 filtres), 32 fois plus coûteux par image, même calcul d'entraînement ;
  COURT   le réseau convolutif seul, 32 fois moins de calcul d'entraînement.
Les taux de départ des quatre recettes sont ceux que le balayage de la clé 40 a choisis (balayage40_choix.json).
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "cnn"
GRAINES = list(range(100, 124))
BRAS = ("INDEPC", "SEUL10", "SEUL", "GRAND", "COURT")
PAR_TRANCHE = 2                      # graines par tranche


def nom(t):
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def course(bras, seed):
    return dict(bras=bras, arch=ARCH, seed=seed, res=RES, taux=None)


def tranches():
    out = {}
    for k in range(len(GRAINES) // PAR_TRANCHE):
        gr = GRAINES[PAR_TRANCHE * k:PAR_TRANCHE * (k + 1)]
        out[f"H{k + 1}"] = [course(b, s) for b in BRAS for s in gr]
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == len(GRAINES) * len(BRAS) == 120
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan46.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v))
