"""Le plan de la clé 47 : python3 plan47.py écrit plan47.json. Tout est sur le réseau convolutif de la clé 40
(32×32×3 → 16×16×16 → 16×16×32 → 8×8×32 → 10), en 32×32, dans le monde des clés 40 à 46, sur les 24 graines de la
clé 46 (100 à 123), dont les courses publiées servent de référence, graine par graine.

HUIT BRAS, sur chaque graine (192 courses ; tranches J1 à J24, une graine et huit courses par tranche)
  K2, K8, K32  la recette du réseau seul pesé à la 10e génération (SEUL10) coupée en 2, 8 ou 32 réseaux, chacun nourri
               de 1/k de son calcul d'entraînement, lus en ensemble ;
  K4P          la même coupée en 4 (K4), puis la distillation de leur ensemble dans un seul réseau (PHOTO) ;
  POPC         la recherche aléatoire convolutive aux réglages faits pour le convolutif, sans curation, suivie du
               choix final de la clé 45 ;
  VEILLEUR     la course de SEUL10 refaite, avec la moyenne de ses poids sur les 30 dernières générations (SWA) et sa
               moyenne mobile (EMA) ;
  TOILE        un seul réseau au taux cyclique (quatre cycles), l'ensemble de ses quatre instantanés ;
  SOUPE        SEUL10 jusqu'à mi-course (le parent), puis quatre enfants nourris chacun du quart du calcul restant, et
               la moyenne de leurs poids (la soupe).
Tous reçoivent le calcul d'entraînement de SEUL10 (384 000 pas d'un petit réseau sur 64 images), sauf PHOTO, qui
s'ajoute à K4 (le calcul du maître, puis celui de l'élève).
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "cnn"
GRAINES = list(range(100, 124))
BRAS = ("K4P", "POPC", "K32", "K8", "K2", "SOUPE", "TOILE", "VEILLEUR")


def nom(t):
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def course(bras, seed):
    return dict(bras=bras, arch=ARCH, seed=seed, res=RES)


def tranches():
    out = {f"J{i + 1}": [course(b, s) for b in BRAS] for i, s in enumerate(GRAINES)}
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == len(GRAINES) * len(BRAS) == 192
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan47.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v))
