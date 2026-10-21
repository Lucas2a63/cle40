"""Le plan de la clé 44 : python3 plan44.py écrit plan44.json. Tout est sur le perceptron, en 32×32, sur les 48
graines de la clé 43 (100 à 147), dans le monde des clés 40 à 43.

LES POPULATIONS (tranches E1 à E20 ; 12 courses par tranche)
  ENS     le système, dont la sélection classe les réseaux selon leur apport à l'ensemble sur l'ancre (nouveau) ;
  ENSR    la même sélection, qui ne copie que les réglages : les poids restent indépendants (nouveau) ;
  INDEP   la recherche aléatoire (avec curation) — refaite : elle doit redonner au bit près les courses de la clé 43,
          avec, en plus, la diversité ;
  SYS     le système entier — refait de même ;
  M32     32 réseaux courts réglés sur deux axes (taux LR0/8 en cosinus, décroissance 3e-2) — refait de même.
Les autres recettes (grand réseau, réseau seul, réseau court réglé) et la recherche aléatoire sans curation ne sont pas
refaites : le verdict reprend leurs courses de la clé 43, sur les mêmes graines. 240 courses.
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "mlp"
GRAINES = list(range(100, 148))
BRAS = ("ENS", "ENSR", "INDEP", "SYS", "M32")
PAR_TRANCHE = 12


def nom(t):
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def course(bras, seed):
    taux = "balayage41_choix.json" if bras == "M32" else None
    return dict(bras=bras, arch=ARCH, seed=seed, res=RES, taux=taux)


def tranches():
    toutes = [course(b, s) for s in GRAINES for b in BRAS]
    out = {f"E{k + 1}": toutes[PAR_TRANCHE * k:PAR_TRANCHE * (k + 1)] for k in range(len(toutes) // PAR_TRANCHE)}
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == 240
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan44.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v[:3]) + ", …")
