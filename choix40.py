"""Le choix des taux de départ des réseaux seuls, à partir du balayage : python3 choix40.py

Règle (écrite avant toute course, comme le pré-enregistrement) : pour chaque bras d'un réseau seul (SEUL, COURT,
GRAND, sur chaque architecture), la note d'un taux est la moyenne des notes de l'ancre aux dix dernières
générations de sa course sur la graine 99 (2 000 images ; jamais le juge). Le taux retenu est celui de la
meilleure note. Si le meilleur taux est au bord de la grille, la grille est prolongée de deux facteurs 2 de ce
côté (deux courses de plus), au plus deux fois par bras ; le prolongement est refait jusqu'à ce que le meilleur
taux soit à l'intérieur ou que les deux prolongements soient faits. SEUL10 prend le taux de SEUL.

Lit resultats/tranche_B*.json (et resultats/tranche_Bx*.json, les prolongements). Écrit :
  - s'il faut prolonger : balayage40_prolongements.json (les courses à faire, pour lancer40.py --taches) ;
  - sinon : balayage40_choix.json (les taux retenus et les grilles), que lit le plan.
N'affiche que les taux retenus ou les prolongements à faire, jamais les notes.
"""
import glob, json, os, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import plan40

PROLONGEMENTS_MAX = 2


def lire():
    notes = {}
    for f in sorted(glob.glob(os.path.join(ICI, "resultats", "tranche_B*.json"))):
        for k, r in json.load(open(f)).items():
            t = r["tache"]
            notes.setdefault((t["bras"], t["arch"]), {})[float(t["mult"])] = r["note_balayage"]
    return notes


if __name__ == "__main__":
    notes = lire()
    grilles = {k: sorted(v) for k, v in plan40.GRILLE.items()}
    afaire, choix, prolonges = [], {}, {}
    for cle, depart in grilles.items():
        n = notes.get(cle, {})
        manque = [m for m in depart if m not in n]
        if manque:
            sys.exit(f"balayage incomplet pour {cle} : {len(manque)} taux sans note")
        ms = sorted(n)
        # combien de prolongements ont déjà été faits de chaque côté (par paires de taux hors de la grille)
        bas = sum(1 for m in ms if m < depart[0]) // 2
        haut = sum(1 for m in ms if m > depart[-1]) // 2
        meilleur = max(ms, key=lambda m: n[m])
        prolonges[f"{cle[0]}_{cle[1]}"] = dict(bas=bas, haut=haut)
        if meilleur == ms[0] and bas < PROLONGEMENTS_MAX:
            afaire += [plan40.balayage(cle[0], cle[1], ms[0] / 2), plan40.balayage(cle[0], cle[1], ms[0] / 4)]
        elif meilleur == ms[-1] and haut < PROLONGEMENTS_MAX:
            afaire += [plan40.balayage(cle[0], cle[1], ms[-1] * 2), plan40.balayage(cle[0], cle[1], ms[-1] * 4)]
        else:
            choix[f"{cle[0]}_{cle[1]}"] = meilleur
            grilles[cle] = ms
    if afaire:
        json.dump(afaire, open(os.path.join(ICI, "balayage40_prolongements.json"), "w"), indent=1)
        print("prolongements à faire :", ", ".join(plan40.nom(t) for t in afaire))
        sys.exit(2)
    for arch in ("cnn", "mlp"):
        choix[f"SEUL10_{arch}"] = choix[f"SEUL_{arch}"]
    choix["grilles"] = {f"{b}_{a}": ms for (b, a), ms in grilles.items()}
    json.dump(choix, open(os.path.join(ICI, "balayage40_choix.json"), "w"), indent=1)
    print("taux retenus (multiples de LR0 = 0,0632) :",
          json.dumps({k: v for k, v in choix.items() if k != "grilles"}))
