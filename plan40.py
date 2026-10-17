"""Le plan de la clé 40 : python3 plan40.py écrit plan40.json.

LE BALAYAGE (graine 99, hors plan ; la note de l'ancre seule)                     tranches B1 à B3
  Pour chaque bras d'un réseau seul (SEUL, COURT, GRAND ; SEUL10 prend le taux de SEUL), une course entière
  par taux de départ de la grille (multiples de LR0 = 0,0632). Les prolongements éventuels (règle dans
  choix40.py) se font après, dans le bac à sable.
LA POPULATION (ne dépend pas du balayage)                                         tranches P1 à P8
  Convolutif, graines 0 à 11 : SYS et INDEP (24 courses, les plus longues), quatre par tranche : P1 à P6.
  Perceptron, graines 0 à 23 : SYS et INDEP (48 courses) : P7 et P8.
LES RÉSEAUX SEULS (après le balayage : leur taux vient de balayage40_choix.json)    tranches R1 à R8
  Convolutif, graines 0 à 11 : SEUL, SEUL10, GRAND, COURT (48 courses) : R1 à R6.
  Perceptron, graines 0 à 23 : SEUL, GRAND, COURT (72 courses) : R7 et R8.
Toutes les courses sont en 32×32 (res = 32). 192 courses du plan, et le balayage (32 courses de départ).
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
RES = 32
GRAINES = {"cnn": list(range(12)), "mlp": list(range(24))}
GRAINE_BALAYAGE = 99
# les grilles de départ du balayage (multiples de LR0), centrées un peu sous les taux retenus à la clé 39
GRILLE = {
    ("SEUL", "cnn"): [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8],
    ("GRAND", "cnn"): [1 / 8, 1 / 4, 1 / 2, 1, 2],
    ("COURT", "cnn"): [1 / 8, 1 / 4, 1 / 2, 1, 2],
    ("SEUL", "mlp"): [1 / 1024, 1 / 512, 1 / 256, 1 / 128, 1 / 64, 1 / 32],
    ("GRAND", "mlp"): [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8],
    ("COURT", "mlp"): [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4],
}


def nom(t):
    if t.get("balayage"):
        return f"B_{t['arch']}_{t['bras']}_{t['mult']:g}"
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def balayage(bras, arch, mult):
    return dict(bras=bras, arch=arch, seed=GRAINE_BALAYAGE, res=RES, balayage=True, mult=mult)


def course(bras, arch, seed):
    return dict(bras=bras, arch=arch, seed=seed, res=RES, balayage=False,
                taux=None if bras in ("SYS", "INDEP") else "balayage40_choix.json")


def tranches():
    out = {}
    B = lambda b, a: [balayage(b, a, m) for m in GRILLE[(b, a)]]
    out["B1"] = B("SEUL", "cnn")[:4]                                    # 4 longues
    out["B2"] = B("SEUL", "cnn")[4:] + B("GRAND", "cnn")                # 2 longues, 5 moyennes
    out["B3"] = B("COURT", "cnn") + B("SEUL", "mlp") + B("GRAND", "mlp") + B("COURT", "mlp")
    pop_cnn = [course(b, "cnn", s) for s in GRAINES["cnn"] for b in ("SYS", "INDEP")]
    for k in range(6):
        out[f"P{k + 1}"] = pop_cnn[4 * k:4 * (k + 1)]
    pop_mlp = [course(b, "mlp", s) for s in GRAINES["mlp"] for b in ("SYS", "INDEP")]
    out["P7"], out["P8"] = pop_mlp[:24], pop_mlp[24:]
    seul_cnn = [course(b, "cnn", s) for s in GRAINES["cnn"] for b in ("SEUL", "SEUL10", "GRAND", "COURT")]
    for k in range(6):
        out[f"R{k + 1}"] = seul_cnn[8 * k:8 * (k + 1)]
    seul_mlp = [course(b, "mlp", s) for s in GRAINES["mlp"] for b in ("SEUL", "GRAND", "COURT")]
    out["R7"], out["R8"] = seul_mlp[:36], seul_mlp[36:]
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms)
    assert sum(len(v) for k, v in out.items() if not k.startswith("B")) == 192
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr, grilles={f"{b}_{a}": g for (b, a), g in GRILLE.items()}),
              open(os.path.join(ICI, "plan40.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k:<3}: {len(v):>2} courses — " + ", ".join(nom(t) for t in v[:3]) + ", …")
