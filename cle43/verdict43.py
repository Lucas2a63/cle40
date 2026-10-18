"""Le verdict de la clé 43, selon la règle écrite dans PREENREGISTREMENT_43.md : python3 verdict43.py

Lit resultats/tranche_E*.json, tranche_L*.json, tranche_C*.json (les 336 courses du plan, perceptron en 32×32, graines
100 à 147), calcule pour chaque course les mesures pré-enregistrées, applique les 17 tests (4 principaux, corrigés
ensemble par la méthode de Holm ; 13 secondaires dont une équivalence) et les deux lectures, et écrit verdicts43.json.

Le « réseau livré » : le mieux noté par l'ancre pour les bras de population (SYS, INDEP, INDEP0, M32), le réseau
lui-même sinon (GRAND, SEUL, COURT2). L'« ensemble » d'une population : la moyenne des probabilités de ses 32 réseaux ;
avec un biais b, la classe qui maximise log(probabilité moyenne) + b·[chat]. Le seuil réglé sur l'ancre (pour un
réseau comme pour un ensemble) : parmi les biais b de la grille du front, ceux qui maximisent la note de l'ancre
(pondérée) sur les ancres réunies des dix dernières générations ; on retient la médiane basse de ces biais (le k-ième
de la liste triée, k = (nombre − 1) // 2), puis on lit la vérité ×30 du juge à ce biais sur le front du juge :
V = w·Rf(b) + (1 − w)·Ao(b), w = 0,9 / 1,87 (le poids du chat dans la vérité ×30 ; calculé ici à partir des
proportions du monde, comme dans le moteur — à 6·10⁻¹⁷ près).
Mesures du réseau livré :
  V_fin  vérité ×30 du juge à la génération 119, au seuil que le réseau se donne ;
  V_cal  la même, au seuil réglé sur l'ancre (fin_livre["ancre"], fin_livre["Rf"], fin_livre["Ao"]) ;
  V_max  (rapportée, sans test) au meilleur seuil du juge ;
  V_vie  au seuil du réseau, en moyenne sur les 16 passages 0, 8, …, 112, 119 ;
  H      note de l'ancre moins vérité du juge, en moyenne sur les 11 passages 40, 48, …, 112, 119.
Mesures de l'ensemble (bras de population) :
  E_fin  vérité ×30 du juge à la génération 119, au seuil par défaut (V30_ens) ;
  E_cal  la même, au seuil réglé sur l'ancre (fin_ens["ancre"], fin_ens["front"]) ;
  E_max  (rapportée, sans test) au meilleur seuil du juge ;
  EG_cal E_cal − E_fin ;
  E_vie  au seuil par défaut, en moyenne sur les 16 passages ;
  H_ens  note de l'ancre de l'ensemble moins sa vérité, en moyenne sur les 11 passages 40, …, 119 (note_ens − V30_ens).
Écart d'un test, par graine : A − B (ou A seul pour les tests à un échantillon), sur les 48 graines.
Règle : Student apparié par graine ; p unilatéral exact (n − 1 degrés de liberté).
  principaux : confirmée si la méthode de Holm, au niveau 0,025 unilatéral sur les 4, la retient ;
               démentie si le p dans le sens contraire est sous 0,025 / 4 ; indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  équivalence (S7) : deux tests unilatéraux, bornes ±0,01, chacun à 0,025 ; démentie si la valeur est
               significativement au-delà d'une borne (p < 0,025) ; indécidable sinon.
  Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
  Plus de deux graines manquantes pour un bras : le test est « incomplet ».
Lectures pré-enregistrées (rapportées sans test supplémentaire) :
  L1 « à seuil égal, l'ensemble de réseaux indépendants est la meilleure capacité mesurée » : E_cal(INDEP) bat
     V_cal(GRAND), V_cal(COURT2), V_cal(SYS), V_cal(SEUL) et E_cal(M32), chacun à p < 0,025 unilatéral ;
  L2 « la curation sert l'ensemble, la sélection le dessert » : E_cal(INDEP) bat E_cal(INDEP0) et E_cal(SYS), chacun
     à p < 0,025 unilatéral.
"""
import glob, json, math, os
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
TARDIF = [g for g in GRILLE if g >= 40]
GRAINES = list(range(100, 148))
ALPHA = 0.025
P_MONDE = np.full(10, (1 - 0.03) / 9); P_MONDE[3] = 0.03
COUT = np.ones(10); COUT[3] = 30.0
W_CHAT = float((P_MONDE * COUT / (P_MONDE * COUT).sum())[3])
POP = ("SYS", "INDEP", "INDEP0", "M32")

# (identifiant, famille, (bras A, mesure A), (bras B, mesure B) ou None, sens attendu, énoncé)
TESTS = [
    ("P1", "principal", ("INDEP", "E_cal"), ("GRAND", "V_cal"), +1,
     "à seuil égal, l'ensemble de la recherche aléatoire bat le grand réseau, au même coût d'usage"),
    ("P2", "principal", ("INDEP", "E_cal"), ("M32", "E_cal"), +1,
     "à seuil égal, l'ensemble de la recherche aléatoire bat l'ensemble de 32 recettes réglées"),
    ("P3", "principal", ("INDEP", "E_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, l'ensemble de la recherche aléatoire bat le réseau livré par le système"),
    ("P4", "principal", ("SYS", "V_cal"), ("SEUL", "V_cal"), +1,
     "à seuil égal, le système bat le réseau seul de la clé 40"),
    ("S1", "secondaire", ("INDEP", "E_cal"), ("COURT2", "V_cal"), +1,
     "à seuil égal, l'ensemble de la recherche aléatoire bat la recette réglée du réseau court"),
    ("S2", "secondaire", ("INDEP", "E_cal"), ("SYS", "E_cal"), +1,
     "à seuil égal, l'ensemble d'une population sans sélection bat celui du système"),
    ("S3", "secondaire", ("INDEP", "E_cal"), ("INDEP0", "E_cal"), +1,
     "à seuil égal, la curation améliore l'ensemble de la recherche aléatoire"),
    ("S4", "secondaire", ("M32", "E_cal"), ("COURT2", "V_cal"), +1,
     "à seuil égal, l'ensemble de 32 recettes réglées bat une seule de ces recettes"),
    ("S5", "secondaire", ("INDEP", "E_cal"), ("INDEP", "V_cal"), +1,
     "à seuil égal, l'ensemble de la recherche aléatoire bat son propre réseau livré"),
    ("S6", "secondaire", ("INDEP0", "E_cal"), ("GRAND", "V_cal"), +1,
     "à seuil égal, même sans curation, l'ensemble de la recherche aléatoire bat le grand réseau"),
    ("S7", "equivalence", ("INDEP", "H_ens"), None, 0,
     "l'ensemble de la recherche aléatoire ne se surestime pas de plus d'un point, ni ne se sous-estime d'autant"),
    ("S8", "secondaire", ("SYS", "H"), ("INDEP", "H_ens"), +1,
     "le réseau livré par le système se surestime plus que l'ensemble de la recherche aléatoire"),
    ("S9", "secondaire", ("INDEP", "E_vie"), ("GRAND", "V_vie"), +1,
     "pendant l'apprentissage, au seuil par défaut, l'ensemble de la recherche aléatoire bat le grand réseau"),
    ("S10", "secondaire", ("INDEP", "E_cal"), ("SEUL", "V_cal"), +1,
     "à seuil égal, l'ensemble de la recherche aléatoire bat le réseau seul de la clé 40"),
    ("S11", "secondaire", ("INDEP", "EG_cal"), None, +1,
     "régler son seuil sur l'ancre améliore l'ensemble de la recherche aléatoire"),
    ("S12", "secondaire", ("M32", "E_cal"), ("SYS", "E_cal"), +1,
     "à seuil égal, l'ensemble de 32 recettes réglées bat celui du système"),
    ("S13", "secondaire", ("SYS", "E_cal"), ("SYS", "V_cal"), +1,
     "à seuil égal, l'ensemble du système bat son réseau livré"),
]
BORNE = 0.01
LECTURES = {
    "L1": (("INDEP", "E_cal"), (("GRAND", "V_cal"), ("COURT2", "V_cal"), ("SYS", "V_cal"), ("SEUL", "V_cal"),
                                ("M32", "E_cal")),
           "à seuil égal, l'ensemble de réseaux indépendants est la meilleure capacité mesurée"),
    "L2": (("INDEP", "E_cal"), (("INDEP0", "E_cal"), ("SYS", "E_cal")),
           "la curation sert l'ensemble, la sélection le dessert"),
}


def lire_front(front, ancre):
    """(V au seuil réglé sur l'ancre, biais retenu, V au meilleur seuil du juge, meilleur biais, V au biais 0)."""
    b = np.array(front["biais"]); V = W_CHAT * np.array(front["Rf"]) + (1 - W_CHAT) * np.array(front["Ao"])
    assert ancre["biais"] == front["biais"], "les grilles du front de l'ancre et du juge diffèrent"
    note = np.array(ancre["note"])
    meilleurs = sorted(np.array(ancre["biais"])[note == note.max()])
    bc = float(meilleurs[(len(meilleurs) - 1) // 2])
    i = int(np.flatnonzero(b == bc)[0])
    i0 = int(np.flatnonzero(b == 0.0)[0])
    return float(V[i]), bc, float(V.max()), float(b[int(V.argmax())]), float(V[i0])


def mesures(r):
    j = {d["g"]: d for d in r["juge"]}
    out = dict(V_fin=j[119]["V30_b"], V_vie=float(np.mean([j[g]["V30_b"] for g in GRILLE])),
               H=float(np.mean([j[g]["note_b"] - j[g]["V30_b"] for g in TARDIF])))
    f = r["fin_livre"]
    out["V_cal"], out["b_cal"], out["V_max"], out["b_max"], _ = lire_front(f, f["ancre"])
    out["G_cal"] = out["V_cal"] - out["V_fin"]
    if "fin_ens" in r:
        e = r["fin_ens"]
        out["E_fin"] = j[119]["V30_ens"]
        out["E_cal"], out["eb_cal"], out["E_max"], out["eb_max"], out["E0"] = lire_front(e["front"], e["ancre"])
        out["EG_cal"] = out["E_cal"] - out["E_fin"]
        out["E_vie"] = float(np.mean([j[g]["V30_ens"] for g in GRILLE]))
        out["H_ens"] = float(np.mean([j[g]["note_ens"] - j[g]["V30_ens"] for g in TARDIF]))
    return out


def charger():
    D = {}
    for motif in ("tranche_E*.json", "tranche_L*.json", "tranche_C*.json"):
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
    a = serie(M, *A)
    if B is None:
        return dict(a), len(GRAINES) - len(a)
    b = serie(M, *B)
    g = sorted(set(a) & set(b))
    manque = max(len(GRAINES) - len(a), len(GRAINES) - len(b))
    return {s: a[s] - b[s] for s in g}, manque


if __name__ == "__main__":
    M = charger()
    res = []
    for tid, fam, A, B, sens, texte in TESTS:
        d, manque = ecarts(M, A, B)
        e = dict(id=tid, famille=fam, A=f"{A[0]}.{A[1]}", B=None if B is None else f"{B[0]}.{B[1]}", enonce=texte,
                 manquantes=manque)
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
        e["p_holm_17"] = a
    lectures = {}
    for lid, (A, contre, texte) in LECTURES.items():
        lignes = {}
        for B in contre:
            d, manque = ecarts(M, A, B)
            if manque > 2 or len(d) < 3:
                lignes[f"{B[0]}.{B[1]}"] = None; continue
            n, moy, sd, t, p_haut, _ = student(list(d.values()))
            lignes[f"{B[0]}.{B[1]}"] = dict(ecart=moy, t=t, p=p_haut)
        tenue = None if any(v is None for v in lignes.values()) else all(v["p"] < ALPHA for v in lignes.values())
        lectures[lid] = dict(enonce=texte, A=f"{A[0]}.{A[1]}", contre=lignes, tenue=tenue)
    json.dump(dict(tests=res, lectures=lectures), open(os.path.join(ICI, "verdicts43.json"), "w"),
              ensure_ascii=False, indent=1)
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        print(f"{e['id']:<4} {e['verdict']:<11} écart {100 * e['ecart']:+7.3f} pt  t = {e['t']:+6.2f}  p = {e['p']:.2g}  "
              f"(Holm sur 17 : {e['p_holm_17']:.2g})  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else ('non' if v['tenue'] is False else 'incomplète')}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
