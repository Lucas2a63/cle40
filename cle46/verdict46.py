"""Le verdict de la clé 46, selon la règle écrite dans PREENREGISTREMENT_46.md :
python3 verdict46.py [--cle43 DOSSIER] [--cle45 DOSSIER]

Lit resultats/tranche_H*.json (les 120 courses du plan : cinq bras sur le réseau convolutif, en 32×32, graines 100 à
123) et, comme références du perceptron, graine par graine, les courses publiées de la clé 45 (la recherche aléatoire
du perceptron suivie du choix final, INDEPC ; par défaut ../cle45/resultats) et de la clé 43 (le réseau court réglé,
COURT2 ; par défaut ../cle43/resultats) — leurs empreintes sont vérifiées ici. Calcule les mesures pré-enregistrées,
applique les 15 tests (4 principaux, corrigés ensemble par la méthode de Holm ; 11 secondaires) et les deux lectures,
et écrit verdicts46.json.

Mesures — le seuil réglé sur l'ancre est celui des clés 42 à 45 : la médiane basse des biais du chat qui maximisent la
note de l'ancre (sur les ancres réunies des dix dernières générations) ; la vérité ×30 du juge
V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87 calculé à partir des proportions du monde :
  pour une règle R du choix final (TOUS, GLOUTON, …) : E_cal, la vérité de l'ensemble de R au seuil réglé sur l'ancre
    (la mesure principale) ; E_fin au biais 0 ; E_max au meilleur biais du juge (un diagnostic) ;
  pour un réseau seul (SEUL10, SEUL, GRAND, COURT, et LIVRE, le réseau livré par la recherche aléatoire — le mieux noté
    par la dernière ancre ; et COURT2 du perceptron) : V_cal, au seuil réglé sur l'ancre de son propre front
    (fin_livre) ; V_fin au biais 0 ; V_max au meilleur biais du juge ;
  AMB, pour l'ensemble des 32 (TOUS) : l'ambiguïté sur le juge (l'écart quadratique moyen des probabilités des réseaux
    à leur moyenne), soit ce que l'ensemble gagne sur le Brier moyen de ses réseaux.
Écart d'un test, par graine : A − B, sur les 24 graines (pour P4, la différence de deux écarts). Règle : Student
apparié ; p unilatéral exact.
  principaux : confirmée si Holm (0,025 unilatéral, sur les 4) la retient ; démentie si le p contraire < 0,025 / 4 ;
               indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens. Plus de deux graines manquantes : « incomplet ».
Lectures (sans test supplémentaire) :
  L1 « sur le convolutif aussi, l'ensemble des 32 garde une avance sur le réseau seul » : P1 et S1 confirmées ;
  L2 « l'architecture compte plus que la population » : P3 et P4 confirmées.
"""
import glob, hashlib, json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRAINES = list(range(100, 124))
ALPHA = 0.025
P_MONDE = np.full(10, (1 - 0.03) / 9); P_MONDE[3] = 0.03
COUT = np.ones(10); COUT[3] = 30.0
W_CHAT = float((P_MONDE * COUT / (P_MONDE * COUT).sum())[3])
REGLES = ("TOUS", "GLOUTON", "ELAGAGE", "MEILLEURS8", "MEILLEURS16", "TETE", "HASARD")
RECETTES = ("SEUL10", "SEUL", "GRAND", "COURT")

# Les courses publiées utilisées comme références (empreintes SHA-256 des fichiers).
EMPREINTES45 = {
    "tranche_F1.json": "0e76104b43c85f0a154b30f9456f8bddd5bfa71aab6b18455c873213e02631a5",
    "tranche_F2.json": "77a9c02aef113aff8b466e7b30de1e40132825e18d5e9450274df8c8c63d2311",
    "tranche_F3.json": "7cad953b7dd4d619ca6eca1908abe4bc97e9e74e74168bbcf50bc844b3186075",
    "tranche_F4.json": "dcd8a603dc3c27f345313505bd0d80c04271cd7a3f649f900a403b34a0c8f9d7",
    "tranche_F5.json": "bacb65fdfb619977087b4724572edb9f099c53c3c4c8a2574ebc3d3d2d5fb6f3",
    "tranche_F6.json": "9bbcb4359aec0c74712629f2b416ccb318cf1dc04b07d1751657f8e7897dc657",
}
EMPREINTES43 = {
    "tranche_C1.json": "bf3810798a2aaa78014ba3316fde1af5aaa173c95d6eaac10da5aaa8d2676461",
    "tranche_C2.json": "cd04f470e1f7313022ebe2567c19086c37f40a5e55e91674998c5d79ca6974f9",
    "tranche_C3.json": "f38bba385e9f19b04dea1d959c92a1a85e1740d505671fb5e2e25df57253891c",
    "tranche_C4.json": "ccc12d4a3adb70f927bb888ef35c202760cbf004f429bf171bea7aedd86a02f9",
    "tranche_L1.json": "49717b90f81cfb1b6e49b8c20f867c4c59971d8bcd8909e13a1d6a20c91cd60a",
    "tranche_L2.json": "0d8a343d3647217404820f1bce07319131abc611cab97c673e8b727149951e3e",
    "tranche_L3.json": "acacd1763201b160de716bbb6bbbc89385bbb1a2dddedb248e3fe5a331fee13d",
    "tranche_L4.json": "7404ae410aaa04873ae51ac4b76281a94bd73e9fd95c4455b5005fe18d1697c0",
    "tranche_L5.json": "ca3820ee6495f71a9bf5c2a09bf17b174c907c77bed8293228a23c45af9dc5de",
    "tranche_L6.json": "f22f860ef5190cc863331545285111b3d61868df66bf430a489ffc392c352e24",
}

# Une mesure : (architecture, bras ou règle, mesure). Un test : (identifiant, famille, A, B ou None, sens, énoncé) ;
# pour P4, A et B sont eux-mêmes des écarts (paires de mesures) : l'écart du test est A − B.
CT = ("cnn", "TOUS", "E_cal")
TESTS = [
    ("P1", "principal", CT, ("cnn", "SEUL10", "V_cal"), +1,
     "sur le convolutif, à seuil égal, l'ensemble des 32 bat le réseau seul pesé à la 10e génération"),
    ("P2", "principal", CT, ("cnn", "GRAND", "V_cal"), +1,
     "sur le convolutif, à seuil égal, l'ensemble des 32 bat le grand réseau (même coût d'usage)"),
    ("P3", "principal", ("cnn", "SEUL10", "V_cal"), ("mlp", "TOUS", "E_cal"), +1,
     "à seuil égal, un seul réseau convolutif (pesé à la 10e) bat l'ensemble des 32 perceptrons (clé 45, même graine)"),
    ("P4", "principal", (("mlp", "TOUS", "E_cal"), ("mlp", "COURT2", "V_cal")), (CT, ("cnn", "SEUL10", "V_cal")), +1,
     "l'avance de l'ensemble sur le réseau seul est plus petite sur le convolutif que sur le perceptron "
     "(perceptron : les 32 moins le réseau court réglé ; convolutif : les 32 moins le réseau seul pesé à la 10e)"),
    ("S1", "secondaire", CT, ("cnn", "SEUL", "V_cal"), +1,
     "sur le convolutif, à seuil égal, l'ensemble des 32 bat le réseau seul pesé dès le départ"),
    ("S2", "secondaire", CT, ("cnn", "COURT", "V_cal"), +1,
     "sur le convolutif, à seuil égal, l'ensemble des 32 bat le réseau court"),
    ("S3", "secondaire", ("cnn", "GLOUTON", "E_cal"), CT, +1,
     "sur le convolutif, à seuil égal, l'ensemble choisi par la sélection gloutonne bat celui des 32"),
    ("S4", "secondaire", ("cnn", "GLOUTON", "E_cal"), ("cnn", "HASARD", "E_cal"), +1,
     "sur le convolutif, à seuil égal, la sélection gloutonne bat autant de réseaux tirés au hasard"),
    ("S5", "secondaire", CT, ("cnn", "MEILLEURS8", "E_cal"), +1,
     "sur le convolutif, à seuil égal, les 32 battent les 8 meilleurs pris un à un"),
    ("S6", "secondaire", CT, ("mlp", "TOUS", "E_cal"), +1,
     "à seuil égal, l'ensemble des 32 réseaux convolutifs bat celui des 32 perceptrons (clé 45, même graine)"),
    ("S7", "secondaire", ("cnn", "SEUL10", "V_cal"), ("cnn", "SEUL", "V_cal"), +1,
     "sur le convolutif, à seuil égal, le réseau seul pesé à la 10e génération bat le même pesé dès le départ"),
    ("S8", "secondaire", ("cnn", "GLOUTON", "E_cal"), ("cnn", "SEUL10", "V_cal"), +1,
     "sur le convolutif, à seuil égal, l'ensemble choisi par la sélection gloutonne bat le réseau seul pesé à la 10e"),
    ("S9", "secondaire", CT, ("cnn", "LIVRE", "V_cal"), +1,
     "sur le convolutif, à seuil égal, l'ensemble des 32 bat le réseau livré par la recherche aléatoire"),
    ("S10", "secondaire", ("cnn", "SEUL10", "V_cal"), ("cnn", "GRAND", "V_cal"), +1,
     "sur le convolutif, à seuil égal, le réseau seul pesé à la 10e bat le grand réseau"),
    ("S11", "secondaire", ("mlp", "TOUS", "AMB"), ("cnn", "TOUS", "AMB"), +1,
     "ce que la moyenne des 32 gagne sur ses réseaux en score de Brier (l'ambiguïté, sur le juge) est plus petit sur le "
     "convolutif que sur le perceptron"),
]
LECTURES = {
    "L1": ("sur le convolutif aussi, l'ensemble des 32 garde une avance sur le réseau seul (P1 et S1 confirmées)",
           ("P1", "S1"), all),
    "L2": ("l'architecture compte plus que la population (P3 et P4 confirmées)", ("P3", "P4"), all),
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
                NB=float(g["nb"]), crit=g["crit"], AMB=g["diversite"]["amb"])


def mesures_reseau(r):
    f = r["fin_livre"]
    V_cal, b, V_max, bm, V0 = lire_front(f, f["ancre"])
    return dict(V_cal=V_cal, b_cal=b, V_max=V_max, V_fin=V0)


def lire_tranches(motif, empreintes=None, dossier=None):
    D = {}
    if empreintes is None:
        fichiers = sorted(glob.glob(motif))
    else:
        fichiers = []
        for nom_f, emp in empreintes.items():
            chemin = os.path.join(dossier, nom_f)
            assert hashlib.sha256(open(chemin, "rb").read()).hexdigest() == emp, f"référence modifiée : {chemin}"
            fichiers.append(chemin)
    for f in fichiers:
        for k, v in json.load(open(f)).items():
            if k in D:
                assert json.dumps(D[k], sort_keys=True) == json.dumps(v, sort_keys=True), f"course en double, différente : {k}"
            D[k] = v
    return D


def charger(d43, d45):
    D = lire_tranches(os.path.join(ICI, "resultats", "tranche_H*.json"))
    R45 = lire_tranches(None, EMPREINTES45, d45)
    R43 = lire_tranches(None, EMPREINTES43, d43)
    return D, R45, R43


def toutes_mesures(D, R45, R43):
    """M[(arch, nom)][graine] = dict des mesures."""
    M = {}
    for s in GRAINES:
        k = f"cnn_INDEPC_s{s}"
        if k in D:
            for nom, g in D[k]["fin_choix"]["regles"].items():
                M.setdefault(("cnn", nom), {})[s] = mesures_regle(g)
            M.setdefault(("cnn", "LIVRE"), {})[s] = mesures_reseau(D[k])
        for b in RECETTES:
            k = f"cnn_{b}_s{s}"
            if k in D:
                M.setdefault(("cnn", b), {})[s] = mesures_reseau(D[k])
        k = f"mlp_INDEPC_s{s}"
        if k in R45:
            for nom, g in R45[k]["fin_choix"]["regles"].items():
                M.setdefault(("mlp", nom), {})[s] = mesures_regle(g)
            M.setdefault(("mlp", "LIVRE"), {})[s] = mesures_reseau(R45[k])
        for b in ("COURT2", "GRAND", "SEUL"):
            k = f"mlp_{b}_s{s}"
            if k in R43:
                M.setdefault(("mlp", b), {})[s] = mesures_reseau(R43[k])
    return M


def serie(M, X):
    """X = (arch, nom, mesure) — ou une paire de telles mesures, dont la série est la différence."""
    if isinstance(X[0], tuple):
        a, b = serie(M, X[0]), serie(M, X[1])
        return {s: a[s] - b[s] for s in a if s in b}
    arch, nom, m = X
    return {s: v[m] for s, v in M.get((arch, nom), {}).items() if m in v}


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


def nom_mesure(X):
    return f"({nom_mesure(X[0])} − {nom_mesure(X[1])})" if isinstance(X[0], tuple) else ".".join(X)


def appliquer(M):
    res = []
    for tid, fam, A, B, sens, texte in TESTS:
        a, b = serie(M, A), serie(M, B)
        g = sorted(set(a) & set(b))
        d = {s: a[s] - b[s] for s in g}
        manque = len(GRAINES) - len(g)
        e = dict(id=tid, famille=fam, A=nom_mesure(A), B=nom_mesure(B), enonce=texte, manquantes=manque)
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


def arg(nom, defaut):
    return sys.argv[sys.argv.index(nom) + 1] if nom in sys.argv else defaut


if __name__ == "__main__":
    D, R45, R43 = charger(arg("--cle43", os.path.join(ICI, "..", "cle43", "resultats")),
                          arg("--cle45", os.path.join(ICI, "..", "cle45", "resultats")))
    import plan46
    attendues = [plan46.nom(t) for v in plan46.tranches().values() for t in v]
    absentes = [k for k in attendues if k not in D]
    en_trop = sorted(set(D) - set(attendues))
    M = toutes_mesures(D, R45, R43)
    res, lectures = appliquer(M)
    json.dump(dict(courses=len(D), absentes=absentes, en_trop=en_trop, tests=res, lectures=lectures),
              open(os.path.join(ICI, "verdicts46.json"), "w"), ensure_ascii=False, indent=1)
    print(f"courses du plan : {len(attendues) - len(absentes)}/{len(attendues)} ; absentes : {absentes or 'aucune'} ; "
          f"en trop : {en_trop or 'aucune'}")
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        print(f"{e['id']:<4} {e['verdict']:<11} écart {100 * e['ecart']:+7.3f}  t = {e['t']:+6.2f}  p = {e['p']:.2g}  "
              f"(Holm sur 15 : {e['p_holm_tous']:.2g})  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else ('non' if v['tenue'] is False else 'incomplète')}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
