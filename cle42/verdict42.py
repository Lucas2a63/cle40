"""Le verdict de la clé 42, selon la règle écrite dans PREENREGISTREMENT_42.md : python3 verdict42.py

Lit resultats/tranche_Y*.json, tranche_Z*.json, tranche_L*.json, tranche_C*.json (les 528 courses du plan,
perceptron en 32×32, graines 48 à 95), calcule pour chaque course les mesures pré-enregistrées, applique les 20 tests
(4 principaux, corrigés ensemble par la méthode de Holm ; 16 secondaires dont une équivalence) et les deux lectures,
et écrit verdicts42.json.

Le « réseau livré » : le mieux noté par l'ancre pour les bras de population (SYS, SYSL, SYSR, INDEP, SANSCUR, M32),
le réseau lui-même sinon. Mesures :
  V_fin  la vérité ×30 du juge pour le réseau livré, à la génération 119, au seuil que le réseau se donne (V30_b) ;
  V_cal  la même au seuil réglé sur l'ancre : parmi les biais b de la grille du front (ajoutés au logit du chat), ceux
         qui maximisent la note de l'ancre du réseau livré sur les ancres réunies des dix dernières générations
         (fin_livre["ancre"]["note"]) ; on retient la médiane basse de ces biais (le k-ième de la liste triée, avec
         k = (nombre − 1) // 2), puis on lit la vérité ×30 du juge à ce biais sur le front du juge :
         V = w·Rf(b) + (1 − w)·Ao(b), w = 0,9 / 1,87 (le poids du chat dans la vérité ×30) ;
  G_cal  V_cal − V_fin ;
  V_vie  la vérité au seuil du réseau, en moyenne sur les 16 passages 0, 8, …, 112, 119 ;
  H      note de l'ancre moins vérité du juge, pour le réseau livré, en moyenne sur les 11 passages 40, 48, …, 119 ;
  E_fin  la vérité ×30 de l'ensemble des 32 réseaux, à la génération 119 ;
  lwd119 le logarithme de la décroissance de la population (moyenne géométrique de ses 32 réseaux), génération 119 ;
  V_max  (rapportée, sans test) la vérité au meilleur seuil du juge, max_b V(b).
Écart d'un test, par graine : A − B, sur les 48 graines.
Règle : Student apparié par graine ; p unilatéral exact (n − 1 degrés de liberté).
  principaux : confirmée si la méthode de Holm, au niveau 0,025 unilatéral sur les 4, la retient ;
               démentie si le p dans le sens contraire est sous 0,025 / 4 ; indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  équivalence (S12) : deux tests unilatéraux, bornes ±0,005, chacun à 0,025 ; démentie si l'écart est
               significativement au-delà d'une borne (p < 0,025) ; indécidable sinon.
  Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
  Plus de deux graines manquantes pour un bras : le test est « incomplet ».
Lectures pré-enregistrées (rapportées sans test supplémentaire) :
  L1 « à seuil égal, le gain du système sur les recettes de la clé 40 disparaît » : sur V_cal, le système ne bat ni le
     grand réseau ni le réseau court de la clé 40 (p ≥ 0,025 unilatéral pour chacun) ;
  L2 « à seuil égal, la population apporte quelque chose sur une recette bien réglée » : sur V_cal, SYSR bat COURT2,
     SEUL2, GRAND2 et M32, chacun à p < 0,025 unilatéral.
"""
import glob, json, math, os
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
TARDIF = [g for g in GRILLE if g >= 40]
GRAINES = list(range(48, 96))
ALPHA = 0.025
P_MONDE = np.full(10, (1 - 0.03) / 9); P_MONDE[3] = 0.03
COUT = np.ones(10); COUT[3] = 30.0
W_CHAT = float((P_MONDE * COUT / (P_MONDE * COUT).sum())[3])

# (identifiant, famille, (bras A, mesure A), (bras B, mesure B), sens attendu, énoncé)
TESTS = [
    ("P1", "principal", ("SYSR", "V_cal"), ("COURT2", "V_cal"), +1,
     "à seuil égal, le système parti de la recette réglée bat la recette réglée du réseau court"),
    ("P2", "principal", ("SYSR", "V_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, partir de la recette réglée améliore le système"),
    ("P3", "principal", ("SYS", "V_cal"), ("INDEP", "V_cal"), +1,
     "à seuil égal, la sélection bat la recherche aléatoire"),
    ("P4", "principal", ("COURT2", "V_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, la recette réglée du réseau court bat le système"),
    ("S1", "secondaire", ("SYSR", "V_cal"), ("M32", "V_cal"), +1,
     "à seuil égal, le système parti de la recette bat la meilleure de 32 recettes réglées"),
    ("S2", "secondaire", ("SYSR", "V_cal"), ("SEUL2", "V_cal"), +1,
     "à seuil égal, le système parti de la recette bat le réseau seul réglé"),
    ("S3", "secondaire", ("SYSR", "V_cal"), ("GRAND2", "V_cal"), +1,
     "à seuil égal, le système parti de la recette bat le grand réseau réglé"),
    ("S4", "secondaire", ("SYSL", "V_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, élargir le génome améliore le système"),
    ("S5", "secondaire", ("SYSL", "V_cal"), ("COURT2", "V_cal"), +1,
     "à seuil égal, le système au génome élargi bat la recette réglée du réseau court"),
    ("S6", "secondaire", ("M32", "V_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, la meilleure de 32 recettes réglées bat le système"),
    ("S7", "secondaire", ("GRAND", "V_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, le grand réseau de la clé 40 bat le système"),
    ("S8", "secondaire", ("COURT", "V_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, le réseau court de la clé 40 bat le système"),
    ("S9", "secondaire", ("SYS", "V_fin"), ("COURT", "V_fin"), +1,
     "au seuil du réseau, le système bat le réseau court de la clé 40 (le gain de la clé 40, sur un troisième jeu de graines)"),
    ("S10", "secondaire", ("COURT", "G_cal"), ("SYS", "G_cal"), +1,
     "régler le seuil sur l'ancre profite plus au réseau court de la clé 40 qu'au système"),
    ("S11", "secondaire", ("SYS", "V_cal"), ("SANSCUR", "V_cal"), +1,
     "à seuil égal, la curation compte : sans elle, le système perd"),
    ("S12", "equivalence", ("M32", "V_cal"), ("COURT2", "V_cal"), 0,
     "à seuil égal, choisir par l'ancre la meilleure de 32 recettes réglées ne change pas la vérité de plus d'un demi-point"),
    ("S13", "secondaire", ("SYSR", "H"), ("COURT2", "H"), +1,
     "le réseau livré par le système parti de la recette se surestime plus que la recette réglée"),
    ("S14", "secondaire", ("SYSL", "lwd119"), ("SYS", "lwd119"), +1,
     "le génome élargi fait monter plus haut la décroissance de la population"),
    ("S15", "secondaire", ("SEUL2", "V_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, le réseau seul réglé bat le système"),
    ("S16", "secondaire", ("COURT2", "V_cal"), ("COURT", "V_cal"), +1,
     "à seuil égal, régler aussi la décroissance améliore le réseau court"),
]
BORNE = 0.005
LECTURES = {
    "L1": ("SYS", ("GRAND", "COURT"), "absence",
           "à seuil égal, le gain du système sur les recettes de la clé 40 disparaît"),
    "L2": ("SYSR", ("COURT2", "SEUL2", "GRAND2", "M32"), "toutes",
           "à seuil égal, la population apporte quelque chose sur une recette bien réglée"),
}


def front_juge(f):
    b = np.array(f["biais"]); V = W_CHAT * np.array(f["Rf"]) + (1 - W_CHAT) * np.array(f["Ao"])
    return b, V


def seuil_ancre(f):
    """Le biais retenu par l'ancre : la médiane basse des biais qui maximisent la note de l'ancre."""
    a = f["ancre"]
    assert a["biais"] == f["biais"], "les grilles du front de l'ancre et du juge diffèrent"
    note = np.array(a["note"])
    meilleurs = sorted(np.array(a["biais"])[note == note.max()])
    return float(meilleurs[(len(meilleurs) - 1) // 2])


def mesures(r):
    j = {d["g"]: d for d in r["juge"]}
    out = dict(V_fin=j[119]["V30_b"], V_vie=float(np.mean([j[g]["V30_b"] for g in GRILLE])),
               H=float(np.mean([j[g]["note_b"] - j[g]["V30_b"] for g in TARDIF])))
    if "V30_ens" in j[119]:
        out["E_fin"] = j[119]["V30_ens"]
    if "lwd" in r["tr"][119]:
        out["lwd119"] = r["tr"][119]["lwd"]
    f = r["fin_livre"]
    b, V = front_juge(f)
    bc = seuil_ancre(f)
    i = int(np.flatnonzero(b == bc)[0])
    out.update(V_cal=float(V[i]), b_cal=bc, V_max=float(V.max()), b_max=float(b[int(V.argmax())]))
    out["G_cal"] = out["V_cal"] - out["V_fin"]
    return out


def charger():
    D = {}
    for motif in ("tranche_Y*.json", "tranche_Z*.json", "tranche_L*.json", "tranche_C*.json"):
        for f in sorted(glob.glob(os.path.join(ICI, "resultats", motif))):
            for k, v in json.load(open(f)).items():
                if k in D:
                    assert json.dumps(D[k]["juge"]) == json.dumps(v["juge"]), f"course en double, différente : {k}"
                D[k] = v
    return {k: mesures(v) for k, v in D.items()}


def serie(M, bras, m):
    return {s: M[f"mlp_{bras}_s{s}"][m] for s in GRAINES if f"mlp_{bras}_s{s}" in M and m in M[f"mlp_{bras}_s{s}"]}


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


def ecarts(M, A, B):
    a, b = serie(M, *A), serie(M, *B)
    g = sorted(set(a) & set(b))
    manque = max(len(GRAINES) - len(a), len(GRAINES) - len(b))
    return {s: a[s] - b[s] for s in g}, manque


if __name__ == "__main__":
    M = charger()
    res = []
    for tid, fam, A, B, sens, texte in TESTS:
        d, manque = ecarts(M, A, B)
        e = dict(id=tid, famille=fam, A=f"{A[0]}.{A[1]}", B=f"{B[0]}.{B[1]}", enonce=texte, manquantes=manque)
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
    for lid, (bras, recettes, mode, texte) in LECTURES.items():
        lignes = {}
        for b in recettes:
            d, manque = ecarts(M, (bras, "V_cal"), (b, "V_cal"))
            if manque > 2 or len(d) < 3:
                lignes[b] = None; continue
            n, moy, sd, t, p_haut, _ = student(list(d.values()))
            lignes[b] = dict(ecart=moy, t=t, p=p_haut)
        if any(v is None for v in lignes.values()):
            tenue = None
        elif mode == "absence":
            tenue = all(v["p"] >= ALPHA for v in lignes.values())
        else:
            tenue = all(v["p"] < ALPHA for v in lignes.values())
        lectures[lid] = dict(enonce=texte, bras=bras, contre=lignes, tenue=tenue)
    json.dump(dict(tests=res, lectures=lectures), open(os.path.join(ICI, "verdicts42.json"), "w"),
              ensure_ascii=False, indent=1)
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        unite = 1 if e["A"].endswith("lwd119") else 100
        print(f"{e['id']:<4} {e['verdict']:<11} écart {unite * e['ecart']:+7.3f}{' pt' if unite == 100 else ' (log)'}  "
              f"t = {e['t']:+6.2f}  p = {e['p']:.2g}  (Holm sur 20 : {e['p_holm_20']:.2g})  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else ('non' if v['tenue'] is False else 'incomplète')}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
