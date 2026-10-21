"""Le verdict de la clé 44, selon la règle écrite dans PREENREGISTREMENT_44.md : python3 verdict44.py [--references DOSSIER]

Lit resultats/tranche_E*.json (les 240 courses du plan, perceptron en 32×32, graines 100 à 147) et, comme références,
les courses publiées de la clé 43 (par défaut ../cle43/resultats ; leurs empreintes sont vérifiées ici) : le grand
réseau de la clé 40 (GRAND) et, pour la vérification d'identité, la recherche aléatoire (INDEP), le système (SYS) et
les 32 recettes réglées (M32). Calcule les mesures pré-enregistrées, applique les 17 tests (4 principaux, corrigés
ensemble par la méthode de Holm ; 13 secondaires dont une équivalence) et les deux lectures, et écrit verdicts44.json.

Identité : les courses INDEP, SYS et M32 refaites doivent redonner au bit près toutes les valeurs des courses publiées
de la clé 43 (hors durée et tâche). Une seule valeur différente rend la clé invalide : les verdicts sont alors écrits,
mais marqués comme tels.

Mesures (celles de la clé 43, dont le seuil réglé sur l'ancre — médiane basse des biais qui maximisent la note de
l'ancre sur les ancres réunies des dix dernières générations — et la vérité ×30 V = w·Rf + (1 − w)·Ao,
w = 0,9 / 1,87 calculé à partir des proportions du monde) :
  V_cal  réseau livré (le mieux noté par l'ancre ; le réseau lui-même pour GRAND), au seuil réglé sur l'ancre ;
  E_cal  ensemble des 32 réseaux d'une population (moyenne des probabilités ; biais sur log p̄(chat)), au seuil réglé ;
  H_ens  note de l'ancre de l'ensemble moins sa vérité, en moyenne sur les 11 passages 40, 48, …, 112, 119 ;
  AMB    l'ambiguïté de la population sur le juge au passage 119, avant la dernière sélection (juge[119]["amb"]) : la
         moyenne, sur les 10 000 images et les 32 réseaux, du carré de l'écart entre les probabilités d'un réseau et
         leur moyenne ;
  LIG    le nombre de lignées (ancêtres distincts des poids) au passage 119 du juge.
Écart d'un test, par graine : A − B (ou A seul), sur les 48 graines. Règle : Student apparié ; p unilatéral exact.
  principaux : confirmée si Holm (0,025 unilatéral, sur les 4) la retient ; démentie si le p contraire < 0,025 / 4 ;
               indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  équivalence (S7) : deux tests unilatéraux, bornes ±0,01, chacun à 0,025 ; démentie si la valeur est
               significativement au-delà d'une borne ; indécidable sinon.
  Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens. Plus de deux graines manquantes : « incomplet ».
Lectures (sans test supplémentaire) :
  L1 « une sélection peut servir l'ensemble » : P1 ou P3 confirmée ;
  L2 « la sélection individuelle ôte sa diversité à l'ensemble, une sélection pour l'ensemble lui en rend une part et
     fait un meilleur ensemble » : P4, S3 et P2 confirmées.
"""
import glob, hashlib, json, math, os, re, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
TARDIF = [g for g in GRILLE if g >= 40]
GRAINES = list(range(100, 148))
ALPHA = 0.025
BORNE = 0.01
P_MONDE = np.full(10, (1 - 0.03) / 9); P_MONDE[3] = 0.03
COUT = np.ones(10); COUT[3] = 30.0
W_CHAT = float((P_MONDE * COUT / (P_MONDE * COUT).sum())[3])
REFAITS = ("INDEP", "SYS", "M32")

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
}

# (identifiant, famille, (bras A, mesure A), (bras B, mesure B) ou None, sens attendu, énoncé)
TESTS = [
    ("P1", "principal", ("ENS", "E_cal"), ("INDEP", "E_cal"), +1,
     "à seuil égal, la sélection pour l'ensemble fait un meilleur ensemble que pas de sélection"),
    ("P2", "principal", ("ENS", "E_cal"), ("SYS", "E_cal"), +1,
     "à seuil égal, la sélection pour l'ensemble fait un meilleur ensemble que la sélection individuelle"),
    ("P3", "principal", ("ENSR", "E_cal"), ("INDEP", "E_cal"), +1,
     "à seuil égal, la sélection pour l'ensemble des seuls réglages fait un meilleur ensemble que pas de sélection"),
    ("P4", "principal", ("INDEP", "AMB"), ("SYS", "AMB"), +1,
     "la sélection individuelle réduit la diversité de la population"),
    ("S1", "secondaire", ("ENS", "E_cal"), ("GRAND", "V_cal"), +1,
     "à seuil égal, l'ensemble de la sélection pour l'ensemble bat le grand réseau, au même coût d'usage"),
    ("S2", "secondaire", ("ENSR", "E_cal"), ("SYS", "E_cal"), +1,
     "à seuil égal, l'ensemble de la sélection des réglages pour l'ensemble bat celui du système"),
    ("S3", "secondaire", ("ENS", "AMB"), ("SYS", "AMB"), +1,
     "la sélection pour l'ensemble garde plus de diversité que la sélection individuelle"),
    ("S4", "secondaire", ("INDEP", "AMB"), ("ENS", "AMB"), +1,
     "mais moins que pas de sélection"),
    ("S5", "secondaire", ("INDEP", "AMB"), ("M32", "AMB"), +1,
     "32 recettes réglées identiques sont moins diverses que la recherche aléatoire"),
    ("S6", "secondaire", ("ENS", "LIG"), ("SYS", "LIG"), +1,
     "la sélection pour l'ensemble garde plus de lignées que la sélection individuelle"),
    ("S7", "equivalence", ("ENS", "H_ens"), None, 0,
     "l'ensemble de la sélection pour l'ensemble ne se surestime pas de plus d'un point, ni ne se sous-estime d'autant"),
    ("S8", "secondaire", ("SYS", "V_cal"), ("ENS", "V_cal"), +1,
     "la sélection individuelle livre un meilleur réseau que la sélection pour l'ensemble"),
    ("S9", "secondaire", ("ENS", "E_cal"), ("ENS", "V_cal"), +1,
     "l'ensemble de la sélection pour l'ensemble bat son propre réseau livré"),
    ("S10", "secondaire", ("ENSR", "E_cal"), ("ENS", "E_cal"), +1,
     "ne copier que les réglages fait un meilleur ensemble que copier aussi les poids"),
    ("S11", "secondaire", ("INDEP", "AMB"), ("ENSR", "AMB"), +1,
     "la sélection des réglages pour l'ensemble réduit la diversité"),
    ("S12", "secondaire", ("ENS", "E_cal"), ("M32", "E_cal"), +1,
     "à seuil égal, l'ensemble de la sélection pour l'ensemble bat celui de 32 recettes réglées"),
    ("S13", "secondaire", ("ENSR", "E_cal"), ("GRAND", "V_cal"), +1,
     "à seuil égal, l'ensemble de la sélection des réglages pour l'ensemble bat le grand réseau"),
]
LECTURES = {
    "L1": ("une sélection peut servir l'ensemble (P1 ou P3 confirmée)", ("P1", "P3"), any),
    "L2": ("la sélection individuelle ôte sa diversité à l'ensemble ; une sélection pour l'ensemble lui en rend une "
           "part et fait un meilleur ensemble (P4, S3 et P2 confirmées)", ("P4", "S3", "P2"), all),
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
    if "fin_ens" in r:
        e = r["fin_ens"]
        out["E_fin"] = j[119]["V30_ens"]
        out["E_cal"], out["eb_cal"], out["E_max"], out["eb_max"], out["E0"] = lire_front(e["front"], e["ancre"])
        out["E_vie"] = float(np.mean([j[g]["V30_ens"] for g in GRILLE]))
        out["H_ens"] = float(np.mean([j[g]["note_ens"] - j[g]["V30_ens"] for g in TARDIF]))
        if "amb" in j[119]:
            out["AMB"] = j[119]["amb"]
            out["DES"] = j[119]["desacc"]
        if "diversite" in e:
            out["AMB_fin"] = e["diversite"]["amb"]
        if "lignees" in j[119]:
            out["LIG"] = float(j[119]["lignees"])
    return out


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
    """Compare toutes les valeurs de la course publiée (hors durée et tâche) à celles de la course refaite."""
    pa = dict(plat({k: v for k, v in ref.items() if k not in ("duree", "tache")}))
    pb = dict(plat({k: v for k, v in nouveau.items() if k not in ("duree", "tache")}))
    n = e = m = 0
    for k, v in pa.items():
        if k not in pb:
            m += 1; continue
        n += 1
        if v != pb[k] or type(v) != type(pb[k]):
            e += 1
    return n, e, m


def charger(dossier_ref):
    D = {}
    for f in sorted(glob.glob(os.path.join(ICI, "resultats", "tranche_E*.json"))):
        for k, v in json.load(open(f)).items():
            if k in D:
                assert json.dumps(D[k]["juge"]) == json.dumps(v["juge"]), f"course en double, différente : {k}"
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
        e.update(n=n, ecart=moy, sd=sd, t=t, ic95=[moy - q * se, moy + q * se], gagne=int(sum(x > 0 for x in d.values())))
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
    par_id = {e["id"]: e for e in res}
    lectures = {}
    for lid, (texte, ids, f) in LECTURES.items():
        vs = [par_id[i]["verdict"] for i in ids]
        tenue = None if "incomplet" in vs else f(v == "confirmée" for v in vs)
        lectures[lid] = dict(enonce=texte, tests={i: v for i, v in zip(ids, vs)}, tenue=tenue)
    return res, lectures


if __name__ == "__main__":
    dref = sys.argv[sys.argv.index("--references") + 1] if "--references" in sys.argv else \
        os.path.join(ICI, "..", "cle43", "resultats")
    D, R = charger(dref)
    ident = dict(courses=0, valeurs=0, ecarts=0, manquantes=0, absentes=[])
    for b in REFAITS:
        for s in GRAINES:
            k = f"mlp_{b}_s{s}"
            if k not in D:
                ident["absentes"].append(k); continue
            n, e, m = identite(D[k], R[k])
            ident["courses"] += 1; ident["valeurs"] += n; ident["ecarts"] += e; ident["manquantes"] += m
    valide = ident["ecarts"] == 0 and ident["manquantes"] == 0
    M = {k: mesures(v) for k, v in D.items()}
    for s in GRAINES:
        M[f"mlp_GRAND_s{s}"] = mesures(R[f"mlp_GRAND_s{s}"])
    res, lectures = appliquer(M)
    json.dump(dict(valide=valide, identite=ident, tests=res, lectures=lectures),
              open(os.path.join(ICI, "verdicts44.json"), "w"), ensure_ascii=False, indent=1)
    print(f"identité avec la clé 43 : {ident['courses']} courses refaites, {ident['valeurs']} valeurs comparées, "
          f"{ident['ecarts']} écarts, {ident['manquantes']} manquantes, {len(ident['absentes'])} absentes"
          + ("" if valide else "  — CLÉ INVALIDE"))
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        print(f"{e['id']:<4} {e['verdict']:<11} écart {100 * e['ecart']:+7.3f}  t = {e['t']:+6.2f}  p = {e['p']:.2g}  "
              f"(Holm sur 17 : {e['p_holm_17']:.2g})  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else ('non' if v['tenue'] is False else 'incomplète')}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
