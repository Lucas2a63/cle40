"""Le verdict de la clé 41, selon la règle écrite dans PREENREGISTREMENT_41.md : python3 verdict41.py

Lit resultats/tranche_A*.json, tranche_D*.json, tranche_W*.json, tranche_M*.json (les 456 courses du plan, perceptron
en 32×32) et reference40.json (les courses publiées de la clé 40, graines 0 à 23, pour SYS, INDEP, SEUL, GRAND,
COURT), calcule pour chaque course les mesures pré-enregistrées, applique les 20 tests (4 principaux, corrigés
ensemble par la méthode de Holm ; 16 secondaires dont une équivalence) et les deux lectures, et écrit verdicts41.json.

Mesures (le « réseau livré » : le mieux noté par l'ancre pour les bras de population — SYS, INDEP, POIDS, REGLAGES,
SANSCUR, M32 —, le réseau lui-même sinon) :
  V_fin  la vérité ×30 du juge pour le réseau livré, à la génération 119 (V30_b) ;
  V_vie  sa moyenne sur les 16 passages 0, 8, …, 112, 119 ;
  H      note de l'ancre moins vérité du juge, pour le réseau livré, en moyenne sur les 11 passages 40, 48, …, 119 ;
  E_fin  la vérité ×30 de l'ensemble des 32 réseaux (moyenne des probabilités) à la génération 119.
Écart d'un test, par graine : A − B. Graines : NEUVES = 24 à 47 ; TOUTES = 0 à 47.
Règle : Student apparié par graine ; p unilatéral exact (n − 1 degrés de liberté).
  principaux : confirmée si la méthode de Holm, au niveau 0,025 unilatéral sur les 4, la retient ;
               démentie si le p dans le sens contraire est sous 0,025 / 4 ; indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  équivalence (S13) : deux tests unilatéraux, bornes ±0,01, chacun à 0,025 ; démentie si l'écart est
               significativement au-delà d'une borne (p < 0,025) ; indécidable sinon.
  Des écarts tous nuls (même course) : t = 0, p = 0,5 dans les deux sens.
  Plus de deux graines manquantes pour un bras : le test est « incomplet ».
Lectures pré-enregistrées (rapportées sans test supplémentaire) :
  L1 « la barre tient sur des graines neuves » : sur les graines 24 à 47, le système bat SEUL, GRAND et COURT,
     chacun à p < 0,025 unilatéral ;
  L2 « la barre renforcée » : sur les graines 0 à 47, le système bat SEUL2, GRAND2, COURT2 et M32, chacun à
     p < 0,025 unilatéral.
"""
import glob, json, math, os
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
TARDIF = [g for g in GRILLE if g >= 40]
NEUVES, TOUTES = list(range(24, 48)), list(range(48))
ALPHA, BORNE = 0.025, 0.01

# (identifiant, famille, (bras A, mesure A), (bras B, mesure B), sens attendu, graines, énoncé)
TESTS = [
    ("P1", "principal", ("SYS", "V_fin"), ("COURT", "V_fin"), +1, NEUVES,
     "réplication : sur des graines neuves, le système bat le réseau court"),
    ("P2", "principal", ("SYS", "V_fin"), ("M32", "V_fin"), +1, TOUTES,
     "le système bat la meilleure de 32 recettes standard, choisie par l'ancre"),
    ("P3", "principal", ("SYS", "V_fin"), ("REGLAGES", "V_fin"), +1, TOUTES,
     "la copie des poids compte : sans elle, le système perd"),
    ("P4", "principal", ("POIDS", "V_fin"), ("REGLAGES", "V_fin"), +1, TOUTES,
     "la copie des poids compte plus que l'évolution des réglages"),
    ("S1", "secondaire", ("SYS", "V_fin"), ("SEUL", "V_fin"), +1, NEUVES,
     "réplication : le système bat le réseau seul qui reçoit tout le calcul"),
    ("S2", "secondaire", ("SYS", "V_fin"), ("GRAND", "V_fin"), +1, NEUVES,
     "réplication : le système bat le grand réseau"),
    ("S3", "secondaire", ("SYS", "V_fin"), ("INDEP", "V_fin"), +1, NEUVES,
     "réplication : la sélection bat la recherche aléatoire"),
    ("S4", "secondaire", ("SYS", "V_fin"), ("SEUL2", "V_fin"), +1, TOUTES,
     "le système bat le réseau seul réglé sur deux axes"),
    ("S5", "secondaire", ("SYS", "V_fin"), ("GRAND2", "V_fin"), +1, TOUTES,
     "le système bat le grand réseau réglé sur deux axes"),
    ("S6", "secondaire", ("SYS", "V_fin"), ("COURT2", "V_fin"), +1, TOUTES,
     "le système bat le réseau court réglé sur deux axes"),
    ("S7", "secondaire", ("COURT2", "V_fin"), ("COURT", "V_fin"), +1, TOUTES,
     "régler aussi la décroissance améliore le réseau court"),
    ("S8", "secondaire", ("SEUL2", "V_fin"), ("SEUL", "V_fin"), +1, TOUTES,
     "régler aussi la décroissance améliore le réseau seul"),
    ("S9", "secondaire", ("M32", "V_fin"), ("COURT2", "V_fin"), +1, TOUTES,
     "choisir par l'ancre la meilleure de 32 recettes standard bat une recette seule"),
    ("S10", "secondaire", ("M32", "E_fin"), ("SYS", "V_fin"), +1, TOUTES,
     "l'ensemble des 32 recettes standard bat le réseau livré par le système"),
    ("S11", "secondaire", ("REGLAGES", "V_fin"), ("INDEP", "V_fin"), +1, TOUTES,
     "l'évolution des réglages, seule, bat la recherche aléatoire"),
    ("S12", "secondaire", ("POIDS", "V_fin"), ("INDEP", "V_fin"), +1, TOUTES,
     "la copie des poids, seule, bat la recherche aléatoire"),
    ("S13", "equivalence", ("SYS", "V_fin"), ("SANSCUR", "V_fin"), 0, TOUTES,
     "la curation ne change pas la vérité finale du système de plus d'un point"),
    ("S14", "secondaire", ("SYS", "H"), ("COURT2", "H"), +1, TOUTES,
     "le réseau livré par le système se surestime plus que la recette standard"),
    ("S15", "secondaire", ("M32", "H"), ("COURT2", "H"), +1, TOUTES,
     "choisir la meilleure de 32 recettes par l'ancre la fait se surestimer"),
    ("S16", "secondaire", ("SYS", "V_fin"), ("POIDS", "V_fin"), +1, TOUTES,
     "l'évolution des réglages compte, en plus de la copie des poids"),
]
LECTURES = {
    "L1": (NEUVES, ("SEUL", "GRAND", "COURT"), "la barre tient sur des graines neuves"),
    "L2": (TOUTES, ("SEUL2", "GRAND2", "COURT2", "M32"), "la barre renforcée"),
}


def mesures(r):
    j = {d["g"]: d for d in r["juge"]}
    out = dict(V_fin=j[119]["V30_b"], V_vie=float(np.mean([j[g]["V30_b"] for g in GRILLE])),
               H=float(np.mean([j[g]["note_b"] - j[g]["V30_b"] for g in TARDIF])))
    if "V30_ens" in j[119]:
        out["E_fin"] = j[119]["V30_ens"]
    return out


def charger():
    D = {}
    for motif in ("tranche_A*.json", "tranche_D*.json", "tranche_W*.json", "tranche_M*.json"):
        for f in sorted(glob.glob(os.path.join(ICI, "resultats", motif))):
            for k, v in json.load(open(f)).items():
                if k in D:
                    assert json.dumps(D[k]["juge"]) == json.dumps(v["juge"]), f"course en double, différente : {k}"
                D[k] = v
    M = {k: mesures(v) for k, v in D.items()}
    R = json.load(open(os.path.join(ICI, "reference40.json")))
    for k, v in R.items():
        assert k not in M, f"une course de référence refaite : {k}"
        M[k] = v
    return M


def serie(M, bras, m, graines):
    return {s: M[f"mlp_{bras}_s{s}"][m] for s in graines if f"mlp_{bras}_s{s}" in M and m in M[f"mlp_{bras}_s{s}"]}


def student(d):
    d = np.asarray(d, float)
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    if se > 0:
        t = d.mean() / se
    else:
        t = 0.0 if d.mean() == 0 else math.copysign(math.inf, d.mean())
    return n, float(d.mean()), float(sd), float(t), float(stats.t.sf(t, n - 1)), float(stats.t.cdf(t, n - 1))


def holm(ps, alpha):
    ordre = sorted(range(len(ps)), key=lambda i: ps[i])
    retenus = set()
    for r, i in enumerate(ordre):
        if ps[i] < alpha / (len(ps) - r):
            retenus.add(i)
        else:
            break
    return retenus


def holm_ajuste(ps):
    ordre = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, courant = [0.0] * len(ps), 0.0
    for r, i in enumerate(ordre):
        courant = max(courant, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = courant
    return adj


def ecarts(M, A, B, graines):
    a, b = serie(M, *A, graines), serie(M, *B, graines)
    g = sorted(set(a) & set(b))
    manque = max(len(graines) - len(a), len(graines) - len(b))
    return {s: a[s] - b[s] for s in g}, manque


if __name__ == "__main__":
    M = charger()
    res = []
    for tid, fam, A, B, sens, graines, texte in TESTS:
        d, manque = ecarts(M, A, B, graines)
        e = dict(id=tid, famille=fam, A=f"{A[0]}.{A[1]}", B=f"{B[0]}.{B[1]}",
                 graines="24-47" if graines == NEUVES else "0-47", enonce=texte, manquantes=manque)
        if manque > 2 or len(d) < 3:
            e.update(verdict="incomplet"); res.append(e); continue
        n, moy, sd, t, p_haut, p_bas = student(list(d.values()))
        se = sd / math.sqrt(n)
        q = float(stats.t.ppf(0.975, n - 1))
        e.update(n=n, ecart=moy, sd=sd, t=t, ic95=[moy - q * se, moy + q * se])
        if fam == "equivalence":
            if se > 0:
                t_bas, t_haut = (moy + BORNE) / se, (moy - BORNE) / se
            else:
                t_bas, t_haut = math.inf, -math.inf
            p1, p2 = float(stats.t.sf(t_bas, n - 1)), float(stats.t.cdf(t_haut, n - 1))
            hors = float(stats.t.sf(t_haut, n - 1)) < ALPHA or float(stats.t.cdf(t_bas, n - 1)) < ALPHA
            e.update(t_bas=t_bas, t_haut=t_haut, p_bas=p1, p_haut=p2,
                     verdict="confirmée" if (p1 < ALPHA and p2 < ALPHA) else ("démentie" if hors else "indécidable"))
            e["p"] = max(p1, p2)
        else:
            p, pc = (p_haut, p_bas) if sens > 0 else (p_bas, p_haut)
            e.update(p=p, p_contraire=pc)
        res.append(e)
    prin = [e for e in res if e["famille"] == "principal" and "p" in e]
    gardes = holm([e["p"] for e in prin], ALPHA)
    for i, e in enumerate(prin):
        e["verdict"] = ("confirmée" if i in gardes else
                        ("démentie" if e["p_contraire"] < ALPHA / len(prin) else "indécidable"))
    for e in res:
        if e["famille"] == "secondaire" and "p" in e:
            e["verdict"] = "confirmée" if e["p"] < ALPHA else ("démentie" if e["p_contraire"] < ALPHA else "indécidable")
    avec_p = [e for e in res if "p" in e]
    for e, a in zip(avec_p, holm_ajuste([e["p"] for e in avec_p])):
        e["p_holm_20"] = a
    lectures = {}
    for lid, (graines, recettes, texte) in LECTURES.items():
        lignes = {}
        for b in recettes:
            d, manque = ecarts(M, ("SYS", "V_fin"), (b, "V_fin"), graines)
            if manque > 2 or len(d) < 3:
                lignes[b] = None; continue
            n, moy, sd, t, p_haut, _ = student(list(d.values()))
            lignes[b] = dict(ecart=moy, t=t, p=p_haut)
        lectures[lid] = dict(enonce=texte, contre=lignes,
                             tenue=all(v is not None and v["p"] < ALPHA for v in lignes.values()))
    json.dump(dict(tests=res, lectures=lectures), open(os.path.join(ICI, "verdicts41.json"), "w"),
              ensure_ascii=False, indent=1)
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        print(f"{e['id']:<4} {e['verdict']:<11} écart {100 * e['ecart']:+6.2f} pt  t = {e['t']:+6.2f}"
              f"  p = {e['p']:.2g}  (Holm sur 20 : {e['p_holm_20']:.2g})  [{e['graines']}]  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else 'non'}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
