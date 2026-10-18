"""Le plan de la clé 43 : python3 plan43.py écrit plan43.json. Tout est sur le perceptron, en 32×32, sur 48 graines
neuves (100 à 147), dans le monde des clés 40 à 42.

LES POPULATIONS ET LEURS ENSEMBLES (tranches E1 à E16 ; 3 graines × 4 bras par tranche)
  SYS     le système entier des clés 36 à 42 ;
  INDEP   la même population, sans sélection : la recherche aléatoire (avec curation) ;
  INDEP0  la recherche aléatoire sans curation ;
  M32     32 réseaux courts réglés sur deux axes (taux LR0/8 en cosinus, décroissance 3e-2), sans sélection ni curation.
  Pour chacune : le réseau livré (le mieux noté par l'ancre) et l'ensemble de ses 32 réseaux.
LE GRAND RÉSEAU (tranches L1 à L6 ; 8 graines par tranche)
  GRAND   le grand réseau de la clé 40 (taux de la clé 40, décroissance 1e-4) — le même coût d'usage qu'un ensemble de
          32 petits réseaux, le même calcul d'entraînement.
LE RÉSEAU SEUL ET LE RÉSEAU COURT (tranches C1 à C4 ; 12 graines × 2 bras par tranche)
  SEUL    le réseau seul de la clé 40 (taux de la clé 40, décroissance 1e-4), qui reçoit tout le calcul ;
  COURT2  le réseau court réglé sur deux axes (LR0/8, 3e-2).
Les réglages viennent de balayage40_choix.json (GRAND, SEUL) et de balayage41_choix.json (COURT2, M32) : il n'y a pas
de nouveau balayage. 336 courses.
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "mlp"
GRAINES = list(range(100, 148))
POP = ("SYS", "INDEP", "INDEP0", "M32")


def nom(t):
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def course(bras, seed):
    taux = None
    if bras in ("SEUL", "GRAND", "COURT"):
        taux = "balayage40_choix.json"
    elif bras in ("SEUL2", "GRAND2", "COURT2", "M32"):
        taux = "balayage41_choix.json"
    return dict(bras=bras, arch=ARCH, seed=seed, res=RES, taux=taux)


def tranches():
    out = {}
    for k in range(16):
        out[f"E{k + 1}"] = [course(b, s) for s in GRAINES[3 * k:3 * k + 3] for b in POP]
    for k in range(6):
        out[f"L{k + 1}"] = [course("GRAND", s) for s in GRAINES[8 * k:8 * k + 8]]
    for k in range(4):
        out[f"C{k + 1}"] = [course(b, s) for s in GRAINES[12 * k:12 * k + 12] for b in ("SEUL", "COURT2")]
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == 336
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan43.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v[:3]) + ", …")
