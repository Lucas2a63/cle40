"""Le plan de la clé 45 : python3 plan45.py écrit plan45.json. Tout est sur le perceptron, en 32×32, sur les 48
graines des clés 43 et 44 (100 à 147), dans le monde des clés 40 à 44.

UN SEUL BRAS (tranches F1 à F12 ; 4 courses par tranche)
  INDEPC  la recherche aléatoire (avec curation, sans sélection), refaite — elle doit redonner au bit près ses
          courses de la clé 43 —, suivie du choix final : sept règles de choix de l'ensemble appliquées à ses 32
          réseaux finaux (echelle45.choix_final). 48 courses.
Le grand réseau, le réseau court réglé et le réseau seul ne sont pas refaits : le verdict reprend leurs courses de la
clé 43, sur les mêmes graines.
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "mlp"
GRAINES = list(range(100, 148))
BRAS = ("INDEPC",)
PAR_TRANCHE = 4


def nom(t):
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def course(bras, seed):
    return dict(bras=bras, arch=ARCH, seed=seed, res=RES, taux=None)


def tranches():
    toutes = [course(b, s) for s in GRAINES for b in BRAS]
    out = {f"F{k + 1}": toutes[PAR_TRANCHE * k:PAR_TRANCHE * (k + 1)] for k in range(len(toutes) // PAR_TRANCHE)}
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == 48
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan45.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v))
