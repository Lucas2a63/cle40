"""Le plan de la clé 48 : python3 plan48.py écrit plan48.json. Tout est sur le réseau convolutif de la clé 40
(32×32×3 → 16×16×16 → 16×16×32 → 8×8×32 → 10), en 32×32, dans le monde des clés 40 à 47.

PARTIE A — un seul réseau recueille-t-il le multiple ? Graines neuves, 124 à 147.
  SEUL10   le réseau seul pesé à la 10e génération, nourri de tout son calcul d'entraînement (384 000 pas) ;
  SEUL_H   le même, nourri de la moitié de ce calcul ;
  PHOTO2   deux réseaux qui se partagent la moitié du calcul (le maître, K2_H), puis la distillation de leur ensemble
           dans un seul réseau nourri de l'autre moitié (l'élève, qui a les poids de départ et les paquets de SEUL_H) ;
  TOILE4   un seul réseau, tout le calcul, au taux cyclique : quatre cycles, un instantané à la fin de chacun ;
  TOILE8   le même, huit cycles.
PARTIE B — le découpage quand le calcul manque. Graines 100 à 123, celles des clés 46 et 47, dont les courses publiées
de la clé 47 (SEUL10, K2, K4, à tout le calcul) servent de référence, graine par graine.
  SEUL_Q, K2_Q, K4_Q   un, deux ou quatre réseaux qui se partagent le quart du calcul de SEUL10 ;
  SEUL_S, K2_S, K4_S   un, deux ou quatre réseaux qui se partagent le seizième du calcul de SEUL10.
24 tranches (J1 à J24) de onze courses : la tranche Ji porte la graine 123 + i de la partie A et la graine 99 + i de la
partie B ; 264 courses en tout.
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "cnn"
GRAINES_A = list(range(124, 148))
GRAINES_B = list(range(100, 124))
BRAS_A = ("PHOTO2", "TOILE8", "TOILE4", "SEUL10", "SEUL_H")
BRAS_B = ("K4_Q", "K2_Q", "SEUL_Q", "K4_S", "K2_S", "SEUL_S")


def nom(t):
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def course(bras, seed):
    return dict(bras=bras, arch=ARCH, seed=seed, res=RES)


def tranches():
    out = {f"J{i + 1}": [course(b, a) for b in BRAS_A] + [course(b, s) for b in BRAS_B]
           for i, (a, s) in enumerate(zip(GRAINES_A, GRAINES_B))}
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == 24 * (len(BRAS_A) + len(BRAS_B)) == 264
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan48.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v))
