"""Le plan de la clé 41 : python3 plan41.py écrit plan41.json. Tout est sur le perceptron, en 32×32.

LE BALAYAGE (graine 99, hors plan ; la note de l'ancre seule)                     tranches G1 à G7
  Pour SEUL2, GRAND2 et COURT2 (la recette standard réglée sur deux axes), une course entière par couple
  (taux de départ, décroissance des poids) de la grille : 5 taux autour du taux retenu à la clé 40 (×1/4 à ×4)
  et 5 décroissances (WD0 = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2). Le couple (taux de la clé 40, WD0) refait la course
  du balayage de la clé 40. Les prolongements éventuels (règle dans choix41.py) se font après.
LA RÉPLICATION (graines neuves 24 à 47 ; ne dépend pas du balayage)               tranches A1 à A6
  SYS, INDEP, SEUL, GRAND, COURT — les bras de la clé 40, aux taux de la clé 40 (balayage40_choix.json).
LA DÉCOMPOSITION (graines 0 à 47 ; ne dépend pas du balayage)                     tranches D1 à D8
  POIDS, REGLAGES, SANSCUR.
LES RECETTES RÉGLÉES SUR DEUX AXES (après le balayage : balayage41_choix.json)     tranches W1 à W6
  SEUL2, GRAND2, COURT2, graines 0 à 47.
LE MEILLEUR DE 32 RECETTES STANDARD (après le balayage ; les réglages de COURT2)   tranches M1 à M3
  M32, graines 0 à 47.
Les références des graines 0 à 23 pour SYS, INDEP, SEUL, GRAND et COURT sont les courses publiées de la clé 40
(reference40.json). 456 courses du plan, et le balayage (75 courses de départ).
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "mlp"
GRAINE_BALAYAGE = 99
NEUVES = list(range(24, 48))
TOUTES = list(range(48))
WD0 = 1e-4
# les taux retenus à la clé 40 (multiples de LR0 = 0,0632), centre des grilles de taux
CENTRE = {"SEUL2": 1 / 1024, "GRAND2": 1 / 16, "COURT2": 1 / 8}
FACTEURS = [1 / 4, 1 / 2, 1, 2, 4]
DECROISSANCES = ["WD0", 3e-4, 1e-3, 3e-3, 1e-2]             # "WD0" : la décroissance des clés 36 à 40


def grille(bras):
    return [(CENTRE[bras] * f, wd) for f in FACTEURS for wd in DECROISSANCES]


def wd_txt(wd):
    return "WD0" if wd == "WD0" else f"{wd:g}"


def nom(t):
    if t.get("balayage"):
        return f"B_{t['bras']}_{t['mult']:g}_{wd_txt(t['wd'])}"
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def balayage(bras, mult, wd):
    return dict(bras=bras, arch=ARCH, seed=GRAINE_BALAYAGE, res=RES, balayage=True, mult=mult, wd=wd)


def course(bras, seed):
    taux = None
    if bras in ("SEUL", "GRAND", "COURT"):
        taux = "balayage40_choix.json"
    elif bras in ("SEUL2", "GRAND2", "COURT2", "M32"):
        taux = "balayage41_choix.json"
    return dict(bras=bras, arch=ARCH, seed=seed, res=RES, balayage=False, taux=taux)


def tranches():
    out = {}
    G = lambda b: [balayage(b, m, wd) for m, wd in grille(b)]
    g2 = G("GRAND2")                                                     # 25 courses moyennes
    out["G1"], out["G2"], out["G3"], out["G4"] = g2[:7], g2[7:13], g2[13:19], g2[19:]
    s2 = G("SEUL2")                                                      # 25 courses moyennes
    out["G5"], out["G6"] = s2[:13], s2[13:]
    out["G7"] = G("COURT2")                                              # 25 courses brèves
    rep = [course(b, s) for s in NEUVES for b in ("SYS", "INDEP", "GRAND", "SEUL", "COURT")]
    for k in range(6):
        out[f"A{k + 1}"] = rep[20 * k:20 * (k + 1)]
    dec = [course(b, s) for s in TOUTES for b in ("POIDS", "REGLAGES", "SANSCUR")]
    for k in range(8):
        out[f"D{k + 1}"] = dec[18 * k:18 * (k + 1)]
    deux = [course(b, s) for s in TOUTES for b in ("GRAND2", "SEUL2", "COURT2")]
    for k in range(6):
        out[f"W{k + 1}"] = deux[24 * k:24 * (k + 1)]
    m32 = [course("M32", s) for s in TOUTES]
    for k in range(3):
        out[f"M{k + 1}"] = m32[16 * k:16 * (k + 1)]
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms)
    assert sum(len(v) for k, v in out.items() if not k.startswith("G")) == 456
    assert sum(len(v) for k, v in out.items() if k.startswith("G")) == 75
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr, grilles={b: grille(b) for b in CENTRE}), open(os.path.join(ICI, "plan41.json"), "w"),
              indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v[:3]) + ", …")
