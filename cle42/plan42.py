"""Le plan de la clé 42 : python3 plan42.py écrit plan42.json. Tout est sur le perceptron, en 32×32, sur 48 graines
neuves (48 à 95), dans le monde des clés 40 et 41.

LES POPULATIONS (tranches Y1 à Y16 ; 3 graines × 5 bras par tranche)
  SYS      le système entier des clés 36 à 41 ;
  SYSL     le système, génome élargi (taux de départ 0,002 à 0,2 ; décroissance de départ 1e-5 à 1e-1 ; bornes 1e-4 à 1
           et 1e-7 à 1) ;
  SYSR     le système, génome parti de la recette réglée du réseau court (taux LR0/8, décroissance 3e-2, ×1/√10 à ×√10) ;
  INDEP    la même population que SYS, sans sélection : la recherche aléatoire ;
  SANSCUR  le système, sans curation.
LE MEILLEUR DE 32 RECETTES RÉGLÉES (tranches Z1 à Z3 ; 16 graines par tranche)
  M32      32 réseaux courts réglés sur deux axes (taux LR0/8, décroissance 3e-2) ; le mieux noté par l'ancre.
LES GRANDS RÉSEAUX (tranches L1 à L6 ; 8 graines × 2 bras par tranche)
  GRAND2   le grand réseau réglé sur deux axes (LR0/4, 3e-2) ; GRAND, celui de la clé 40 (taux de la clé 40, 1e-4).
LE RÉSEAU SEUL ET LES RÉSEAUX COURTS (tranches C1 à C4 ; 12 graines × 3 bras par tranche)
  SEUL2    le réseau seul réglé sur deux axes (LR0/128, 3e-2) ; COURT2, le réseau court réglé (LR0/8, 3e-2) ; COURT,
           le réseau court de la clé 40 (taux de la clé 40, 1e-4).
Les réglages viennent de balayage40_choix.json (GRAND, COURT) et de balayage41_choix.json (SEUL2, GRAND2, COURT2, M32) :
il n'y a pas de nouveau balayage. 528 courses.
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES, ARCH = 32, "mlp"
GRAINES = list(range(48, 96))
POP = ("SYS", "SYSL", "SYSR", "INDEP", "SANSCUR")


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
        out[f"Y{k + 1}"] = [course(b, s) for s in GRAINES[3 * k:3 * k + 3] for b in POP]
    for k in range(3):
        out[f"Z{k + 1}"] = [course("M32", s) for s in GRAINES[16 * k:16 * k + 16]]
    for k in range(6):
        out[f"L{k + 1}"] = [course(b, s) for s in GRAINES[8 * k:8 * k + 8] for b in ("GRAND2", "GRAND")]
    for k in range(4):
        out[f"C{k + 1}"] = [course(b, s) for s in GRAINES[12 * k:12 * k + 12] for b in ("SEUL2", "COURT2", "COURT")]
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == 528
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan42.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v[:3]) + ", …")
