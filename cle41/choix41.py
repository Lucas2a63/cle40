"""Le choix du taux de départ et de la décroissance des poids des recettes réglées sur deux axes (SEUL2, GRAND2,
COURT2), à partir du balayage : python3 choix41.py

Règle (écrite avant toute course, comme le pré-enregistrement) : la note d'un couple (taux, décroissance) est la
moyenne des notes de l'ancre aux dix dernières générations de sa course sur la graine 99 (2 000 images ; jamais le
juge). Le couple retenu est celui de la meilleure note. S'il est au bord de la grille dans une dimension — à
décroissance égale, aucun taux plus petit (ou plus grand) n'a été essayé ; ou, à taux égal, aucune décroissance
plus petite (ou plus grande) —, on essaie deux valeurs de plus dans ce sens (taux : ×1/2 et ×1/4, ou ×2 et ×4 ;
décroissance : les deux crans suivants de la suite 1e-5, 3e-5, WD0 = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
3e-1), l'autre réglage fixé à sa valeur retenue ; au plus deux fois par dimension et par sens, pour chaque bras ;
au bord des deux dimensions, on prolonge les deux. M32 prend les réglages de COURT2.

Lit resultats/tranche_G*.json (et resultats/tranche_Gx*.json, les prolongements). Écrit :
  - s'il faut prolonger : balayage41_prolongements.json (les courses à faire, pour lancer41.py --taches) ;
  - sinon : balayage41_choix.json (les réglages retenus et les grilles essayées), que lit le plan.
N'affiche que les réglages retenus ou les prolongements à faire, jamais les notes.
"""
import glob, json, os, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import plan41

PROLONGEMENTS_MAX = 2
SUITE_WD = [1e-5, 3e-5, "WD0", 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def rang_wd(w):
    return SUITE_WD.index(w)


def cle_wd(w):
    return "WD0" if w == "WD0" else float(w)


def lire():
    notes = {}
    for f in sorted(glob.glob(os.path.join(ICI, "resultats", "tranche_G*.json"))):
        for k, r in json.load(open(f)).items():
            t = r["tache"]
            notes.setdefault(t["bras"], {})[(float(t["mult"]), cle_wd(t["wd"]))] = r["note_balayage"]
    return notes


if __name__ == "__main__":
    notes = lire()
    afaire, choix, essais = [], {}, {}
    for bras in plan41.CENTRE:
        depart = [(float(m), cle_wd(w)) for m, w in plan41.grille(bras)]
        n = notes.get(bras, {})
        manque = [c for c in depart if c not in n]
        if manque:
            sys.exit(f"balayage incomplet pour {bras} : {len(manque)} couples sans note")
        mults0 = sorted({m for m, _ in depart}); wds0 = sorted({rang_wd(w) for _, w in depart})
        (m, w) = max(n, key=lambda c: n[c])
        # prolongements déjà faits, par dimension et par sens (paires de valeurs hors de la grille de départ)
        bas_m = sum(1 for (mm, ww) in n if mm < mults0[0]) // 2
        haut_m = sum(1 for (mm, ww) in n if mm > mults0[-1]) // 2
        bas_w = sum(1 for (mm, ww) in n if rang_wd(ww) < wds0[0]) // 2
        haut_w = sum(1 for (mm, ww) in n if rang_wd(ww) > wds0[-1]) // 2
        ligne = sorted(mm for (mm, ww) in n if ww == w)                       # les taux essayés à cette décroissance
        colonne = sorted(rang_wd(ww) for (mm, ww) in n if mm == m)           # les décroissances essayées à ce taux
        nouv = []
        if m == ligne[0] and bas_m < PROLONGEMENTS_MAX:
            nouv += [plan41.balayage(bras, m / 2, w), plan41.balayage(bras, m / 4, w)]
        elif m == ligne[-1] and haut_m < PROLONGEMENTS_MAX:
            nouv += [plan41.balayage(bras, m * 2, w), plan41.balayage(bras, m * 4, w)]
        r = rang_wd(w)
        if r == colonne[0] and bas_w < PROLONGEMENTS_MAX and r >= 2:
            nouv += [plan41.balayage(bras, m, SUITE_WD[r - 1]), plan41.balayage(bras, m, SUITE_WD[r - 2])]
        elif r == colonne[-1] and haut_w < PROLONGEMENTS_MAX and r + 2 < len(SUITE_WD):
            nouv += [plan41.balayage(bras, m, SUITE_WD[r + 1]), plan41.balayage(bras, m, SUITE_WD[r + 2])]
        nouv = [t for t in nouv if (float(t["mult"]), cle_wd(t["wd"])) not in n]
        if nouv:
            afaire += nouv
        else:
            choix[bras] = dict(mult=m, wd=w)
        essais[bras] = sorted([[mm, ww] for (mm, ww) in n], key=lambda c: (c[0], rang_wd(c[1])))
    if afaire:
        json.dump(afaire, open(os.path.join(ICI, "balayage41_prolongements.json"), "w"), indent=1)
        print("prolongements à faire :", ", ".join(plan41.nom(t) for t in afaire))
        sys.exit(2)
    choix["M32"] = dict(choix["COURT2"])
    choix["grilles"] = essais
    json.dump(choix, open(os.path.join(ICI, "balayage41_choix.json"), "w"), indent=1)
    print("réglages retenus (taux en multiples de LR0 = 0,0632) :",
          json.dumps({k: v for k, v in choix.items() if k != "grilles"}))
