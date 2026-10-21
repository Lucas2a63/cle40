"""Le verdict de la clé 45, selon la règle écrite dans PREENREGISTREMENT_45.md : python3 verdict45.py [--references DOSSIER]

Lit resultats/tranche_F*.json (les 48 courses du plan : la recherche aléatoire INDEPC, perceptron en 32×32, graines
100 à 147, suivie du choix final) et, comme références, les courses publiées de la clé 43 (par défaut
../cle43/resultats ; leurs empreintes sont vérifiées ici) : la recherche aléatoire (INDEP), pour l'identité, et le
grand réseau de la clé 40 (GRAND) — l'analyse descriptive y lit aussi le réseau court réglé et le réseau seul. Calcule les mesures pré-enregistrées de chaque règle de choix, applique les 13 tests
(4 principaux, corrigés ensemble par la méthode de Holm ; 9 secondaires) et les deux lectures, et écrit
verdicts45.json.

Identité : chaque course INDEPC doit redonner au bit près toutes les valeurs de la course INDEP publiée de la clé 43,
de même graine (hors durée et tâche), et sa règle TOUS doit redonner au bit près le front, le front de l'ancre et la
diversité de son fin_ens. Une seule valeur différente rend la clé invalide : les verdicts sont alors écrits, mais
marqués comme tels.

Mesures, pour chaque règle R (fin_choix["regles"][R]) — le seuil réglé sur l'ancre est celui des clés 42 à 44 : la
médiane basse des biais du chat qui maximisent la note de l'ancre (sur les ancres réunies des dix dernières
générations) ; la vérité ×30 du juge V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87 calculé à partir des proportions du monde :
  E_cal  la vérité de l'ensemble de R au seuil réglé sur l'ancre (la mesure principale) ;
  E_fin  la même au biais 0 (le seuil par défaut) ; E_max au meilleur biais du juge (un diagnostic) ;
  H0     la note de l'ancre de l'ensemble de R au biais 0 (sur les ancres réunies) moins sa vérité au biais 0 ;
  NB16   16 moins le nombre de réseaux distincts de R.
Pour le grand réseau : V_cal, au seuil réglé sur l'ancre de son propre front (fin_livre).
Écart d'un test, par graine : A − B (ou A seul), sur les 48 graines. Règle : Student apparié ; p unilatéral exact.
  principaux : confirmée si Holm (0,025 unilatéral, sur les 4) la retient ; démentie si le p contraire < 0,025 / 4 ;
               indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens. Plus de deux graines manquantes : « incomplet ».
Lectures (sans test supplémentaire) :
  L1 « une sélection faite une seule fois, à la fin, sert l'ensemble » : P1 et P2 confirmées ;
  L2 « les meilleurs réseaux pris un à un ne font pas le meilleur ensemble » : P3 et P4 confirmées.
"""
import glob, hashlib, json, math, os, re, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRAINES = list(range(100, 148))
ALPHA = 0.025
P_MONDE = np.full(10, (1 - 0.03) / 9); P_MONDE[3] = 0.03
COUT = np.ones(10); COUT[3] = 30.0
W_CHAT = float((P_MONDE * COUT / (P_MONDE * COUT).sum())[3])
REGLES = ("TOUS", "GLOUTON", "ELAGAGE", "MEILLEURS8", "MEILLEURS16", "TETE", "HASARD")

# Les courses publiées de la clé 43 utilisées comme références (empreintes SHA-256 des fichiers).
EMPREINTES43 = {
    "tranche_E1.json": "c375a12182ea3f8ba710965bb86c15c757fccc124cefddb77bb355e11b54b2b4",
    "tranche_E2.json": "b0f6937b7e248e4f6f23bdccacc9908693ec063c34ff911dc5f3afd9b396908b",
    "tranche_E3.json": "5070e5bd81f7cb756cec0314389759ade8253e8abdcf62371841ebe9ba2a2938",
    "tranche_E4.json": "d066f8f03855dae89c500c4959eb33be202d3929df0e96585aaa944d66ffbe24",
    "tranche_E5.json": "d459179eaa0533b0040508d493e238013c5de596eb80aaf065f43b6f36521eb9",
    "tranche_E6.json": "9970bf2e6172c8cfeb33ebdd78583d7b9cf0281be8863cd9cb7084567e19e21d",
    "tranche_E7.json": "cc90c7883f763a74a5824e039eb1e11176f927bf7d66ec74ece8eab8ce224687",
    "tranche_E8.json": "3c20fb98dc61322037f37a74580abef3579f8b321b375b9a36c51466e879a4b4",
    "tranche_E9.json": "817ca79a6f29a9be254f53e548d747dd3f6f97de4e2bf4fd1865904bdaceed89",
    "tranche_E10.json": "fe983a8206cd136347a2c9ac970e972a061da1a360c636e6d207df3810021022",
    "tranche_E11.json": "e5808e2b10fd7c696cdcdbf6dce660f026b8918292409e6d192db7186b57172f",
    "tranche_E12.json": "a91fe9571dc2d9123e94c69fc4acd733b132c3d4914e42dc418ea0d394c813a3",
    "tranche_E13.json": "c7803b28f359ea590974026a16dcba32d438125043ae521ed04936576fb18aa0",
    "tranche_E14.json": "da8f2f261ea731e0494d668b9a15ad1b7d38ffca34eb3f2b6f460eb1387ffaa8",
    "tranche_E15.json": "b692fab037342f00ddae1245e563748199053fda5fd18e16b094d8ba859f7966",
    "tranche_E16.json": "83a520705fa4620d4d85ad12f70daa477c99698298aad1331d7a6f8fbb1a3da4",
    "tranche_L1.json": "49717b90f81cfb1b6e49b8c20f867c4c59971d8bcd8909e13a1d6a20c91cd60a",
    "tranche_L2.json": "0d8a343d3647217404820f1bce07319131abc611cab97c673e8b727149951e3e",
    "tranche_L3.json": "acacd1763201b160de716bbb6bbbc89385bbb1a2dddedb248e3fe5a331fee13d",
    "tranche_L4.json": "7404ae410aaa04873ae51ac4b76281a94bd73e9fd95c4455b5005fe18d1697c0",
    "tranche_L5.json": "ca3820ee6495f71a9bf5c2a09bf17b174c907c77bed8293228a23c45af9dc5de",
    "tranche_L6.json": "f22f860ef5190cc863331545285111b3d61868df66bf430a489ffc392c352e24",
    "tranche_C1.json": "bf3810798a2aaa78014ba3316fde1af5aaa173c95d6eaac10da5aaa8d2676461",
    "tranche_C2.json": "cd04f470e1f7313022ebe2567c19086c37f40a5e55e91674998c5d79ca6974f9",
    "tranche_C3.json": "f38bba385e9f19b04dea1d959c92a1a85e1740d505671fb5e2e25df57253891c",
    "tranche_C4.json": "ccc12d4a3adb70f927bb888ef35c202760cbf004f429bf171bea7aedd86a02f9",
}

# (identifiant, famille, (bras A, mesure A), (bras B, mesure B) ou None, sens attendu, énoncé)
TESTS = [
    ("P1", "principal", ("GLOUTON", "E_cal"), ("TOUS", "E_cal"), +1,
     "à seuil égal, l'ensemble choisi par la sélection gloutonne bat celui des 32 réseaux"),
    ("P2", "principal", ("GLOUTON", "E_cal"), ("HASARD", "E_cal"), +1,
     "à seuil égal, la sélection gloutonne bat autant de réseaux tirés au hasard (même coût d'usage)"),
    ("P3", "principal", ("TOUS", "E_cal"), ("MEILLEURS8", "E_cal"), +1,
     "à seuil égal, les 32 réseaux battent les 8 meilleurs pris un à un"),
    ("P4", "principal", ("GLOUTON", "E_cal"), ("TETE", "E_cal"), +1,
     "à seuil égal, la sélection gloutonne bat autant de réseaux choisis un à un (les d meilleurs)"),
    ("S1", "secondaire", ("GLOUTON", "E_cal"), ("GRAND", "V_cal"), +1,
     "à seuil égal, l'ensemble choisi par la sélection gloutonne bat le grand réseau"),
    ("S2", "secondaire", ("ELAGAGE", "E_cal"), ("TOUS", "E_cal"), +1,
     "à seuil égal, l'élagage bat les 32 réseaux"),
    ("S3", "secondaire", ("GLOUTON", "H0"), ("TOUS", "H0"), +1,
     "l'ancre surestime l'ensemble choisi sur elle plus que celui des 32 réseaux"),
    ("S4", "secondaire", ("MEILLEURS16", "E_cal"), ("MEILLEURS8", "E_cal"), +1,
     "à seuil égal, les 16 meilleurs battent les 8 meilleurs"),
    ("S5", "secondaire", ("TOUS", "E_cal"), ("HASARD", "E_cal"), +1,
     "à seuil égal, les 32 réseaux battent les d réseaux tirés au hasard"),
    ("S6", "secondaire", ("GLOUTON", "E_fin"), ("TOUS", "E_fin"), +1,
     "au seuil par défaut, l'ensemble choisi par la sélection gloutonne bat celui des 32 réseaux"),
    ("S7", "secondaire", ("GLOUTON", "NB16"), None, +1,
     "la sélection gloutonne retient moins de 16 réseaux distincts"),
    ("S8", "secondaire", ("GLOUTON", "E_cal"), ("MEILLEURS16", "E_cal"), +1,
     "à seuil égal, la sélection gloutonne bat les 16 meilleurs"),
    ("S9", "secondaire", ("TOUS", "E_cal"), ("MEILLEURS16", "E_cal"), +1,
     "à seuil égal, les 32 réseaux battent les 16 meilleurs"),
]
LECTURES = {
    "L1": ("une sélection faite une seule fois, à la fin, sert l'ensemble (P1 et P2 confirmées)", ("P1", "P2"), all),
    "L2": ("les meilleurs réseaux pris un à un ne font pas le meilleur ensemble (P3 et P4 confirmées)", ("P3", "P4"), all),
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


def mesures_regle(g):
    E_cal, eb, E_max, ebm, E0 = lire_front(g["front"], g["ancre"])
    i0 = g["ancre"]["biais"].index(0.0)
    return dict(E_cal=E_cal, eb_cal=eb, E_max=E_max, eb_max=ebm, E_fin=E0, H0=g["ancre"]["note"][i0] - E0,
                NB=float(g["nb"]), NB16=16.0 - g["nb"], crit=g["crit"], AMB=g["diversite"]["amb"])


def mesures_grand(r):
    f = r["fin_livre"]
    V_cal, b, V_max, bm, V0 = lire_front(f, f["ancre"])
    return dict(V_cal=V_cal, b_cal=b, V_max=V_max, V_fin=V0)


def plat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from plat(v, f"{p}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from plat(v, f"{p}[{i}]")
    else:
        yield p, x


def identite(nouveau, ref):
    """Compare toutes les valeurs de la course publiée (hors durée et tâche) à celles de la course refaite ; puis la
    règle TOUS à fin_ens (front, front de l'ancre, diversité)."""
    pa = dict(plat({k: v for k, v in ref.items() if k not in ("duree", "tache")}))
    pb = dict(plat({k: v for k, v in nouveau.items() if k not in ("duree", "tache")}))
    n = e = m = 0
    for k, v in pa.items():
        if k not in pb:
            m += 1; continue
        n += 1
        if v != pb[k] or type(v) != type(pb[k]):
            e += 1
    T, F = nouveau["fin_choix"]["regles"]["TOUS"], nouveau["fin_ens"]
    for a, b in ((T["front"], F["front"]), (T["ancre"]["note"], F["ancre"]["note"]), (T["diversite"], F["diversite"])):
        qa, qb = dict(plat(a)), dict(plat(b))
        for k in qb:
            if k not in qa:
                m += 1; continue
            n += 1
            if qa[k] != qb[k] or type(qa[k]) != type(qb[k]):
                e += 1
    return n, e, m


def charger(dossier_ref):
    D = {}
    for f in sorted(glob.glob(os.path.join(ICI, "resultats", "tranche_F*.json"))):
        for k, v in json.load(open(f)).items():
            if k in D:
                assert json.dumps(D[k], sort_keys=True) == json.dumps(v, sort_keys=True), f"course en double, différente : {k}"
            D[k] = v
    R = {}
    for nom_f, emp in EMPREINTES43.items():
        chemin = os.path.join(dossier_ref, nom_f)
        assert hashlib.sha256(open(chemin, "rb").read()).hexdigest() == emp, f"référence modifiée : {chemin}"
        R.update(json.load(open(chemin)))
    return D, R


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


def appliquer(M):
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
        e.update(n=n, ecart=moy, sd=sd, t=t, ic95=[moy - q * se, moy + q * se], gagne=int(sum(x > 0 for x in d.values())),
                 perd=int(sum(x < 0 for x in d.values())))
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
        e["p_holm_tous"] = a
    par_id = {e["id"]: e for e in res}
    lectures = {}
    for lid, (texte, ids, f) in LECTURES.items():
        vs = [par_id[i]["verdict"] for i in ids]
        tenue = None if "incomplet" in vs else f(v == "confirmée" for v in vs)
        lectures[lid] = dict(enonce=texte, tests={i: v for i, v in zip(ids, vs)}, tenue=tenue)
    return res, lectures


def toutes_mesures(D, R):
    M = {}
    for s in GRAINES:
        k = f"mlp_INDEPC_s{s}"
        if k in D:
            for nom, g in D[k]["fin_choix"]["regles"].items():
                M[f"mlp_{nom}_s{s}"] = mesures_regle(g)
        M[f"mlp_GRAND_s{s}"] = mesures_grand(R[f"mlp_GRAND_s{s}"])
    return M


if __name__ == "__main__":
    dref = sys.argv[sys.argv.index("--references") + 1] if "--references" in sys.argv else \
        os.path.join(ICI, "..", "cle43", "resultats")
    D, R = charger(dref)
    ident = dict(courses=0, valeurs=0, ecarts=0, manquantes=0, absentes=[])
    for s in GRAINES:
        k = f"mlp_INDEPC_s{s}"
        if k not in D:
            ident["absentes"].append(k); continue
        n, e, m = identite(D[k], R[f"mlp_INDEP_s{s}"])
        ident["courses"] += 1; ident["valeurs"] += n; ident["ecarts"] += e; ident["manquantes"] += m
    valide = ident["ecarts"] == 0 and ident["manquantes"] == 0
    M = toutes_mesures(D, R)
    res, lectures = appliquer(M)
    json.dump(dict(valide=valide, identite=ident, tests=res, lectures=lectures),
              open(os.path.join(ICI, "verdicts45.json"), "w"), ensure_ascii=False, indent=1)
    print(f"identité avec la clé 43 : {ident['courses']} courses refaites, {ident['valeurs']} valeurs comparées, "
          f"{ident['ecarts']} écarts, {ident['manquantes']} manquantes, {len(ident['absentes'])} absentes"
          + ("" if valide else "  — CLÉ INVALIDE"))
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        u = 1 if e["A"].endswith("NB16") else 100
        print(f"{e['id']:<4} {e['verdict']:<11} écart {u * e['ecart']:+7.3f}  t = {e['t']:+6.2f}  p = {e['p']:.2g}  "
              f"(Holm sur 13 : {e['p_holm_tous']:.2g})  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else ('non' if v['tenue'] is False else 'incomplète')}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
