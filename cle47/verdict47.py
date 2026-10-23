"""Le verdict de la clé 47, selon la règle écrite dans PREENREGISTREMENT_47.md :
python3 verdict47.py [--cle46 DOSSIER] [--resultats DOSSIER]

Lit resultats/tranche_J*.json (les 192 courses du plan : huit bras sur le réseau convolutif, en 32×32, graines 100 à
123) et, comme références, graine par graine, les courses publiées de la clé 46 (par défaut ../cle46/resultats ; leurs
empreintes sont vérifiées ici) : la recherche aléatoire convolutive (INDEPC) et le réseau seul pesé à la 10e
(SEUL10). Calcule les mesures pré-enregistrées, applique les 17 tests (4 principaux, corrigés ensemble par la méthode
de Holm ; 13 secondaires) et les deux lectures, vérifie que le réseau du VEILLEUR refait au bit près la course SEUL10
de la clé 46 et que le parent de la SOUPE en refait la première moitié, et écrit verdicts47.json.

Mesures — le seuil réglé sur l'ancre est celui des clés 42 à 46 : la médiane basse des biais du chat qui maximisent la
note de l'ancre (sur les ancres réunies des dix dernières générations) ; la vérité ×30 du juge
V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87 calculé à partir des proportions du monde :
  pour un ensemble (K2, K4, K8, K32, l'ensemble des instantanés de la TOILE, celui des enfants de la SOUPE, et les
    règles du choix final de POPC et d'INDEPC) : E_cal, sa vérité au seuil réglé sur son propre front de l'ancre ;
    E_fin au biais 0 ; E_max au meilleur biais du juge (un diagnostic) ; AMB, l'ambiguïté sur le juge ;
  pour un réseau seul (SEUL10 : le réseau du VEILLEUR ; SWA, EMA ; PHOTO ; SOUPE, la soupe, et GOURMANDE ; le dernier
    instantané de la TOILE ; la moyenne des poids des réseaux de K4, MOY_K4 — la soupe sans ancêtre commun) : V_cal, au
    seuil réglé sur son propre front de l'ancre ; V_fin ; V_max ;
  pour les réseaux d'un ensemble pris seuls : V_cal de chacun, et leur moyenne (MEMBRES).
Écart d'un test, par graine : A − B, sur les 24 graines. Règle : Student apparié ; p unilatéral exact.
  principaux : confirmée si Holm (0,025 unilatéral, sur les 4) la retient ; démentie si le p contraire < 0,025 / 4 ;
               indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens. Plus de deux graines manquantes : « incomplet ».
Lectures (sans test supplémentaire) :
  L1 « à calcul égal, la meilleure recette coupée en plusieurs réseaux, lus en ensemble, bat le réseau seul » :
     au moins une de P1, S1, S2 confirmée ;
  L2 « à calcul égal et au même coût d'usage, un seul réseau recueille le multiple » : au moins une de P3, S6, S7
     confirmée.
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

# Les courses publiées de la clé 46 utilisées comme références (empreintes SHA-256 des fichiers).
EMPREINTES46 = {
    "tranche_H1.json": "d5eb3acf851d68d07e69ff5e908570c1ae64f34c148dc113527a4639b14309cc",
    "tranche_H2.json": "682cee84fe491e217563fbb42b8101a3e634a626b46eba1e123ab2b53638a304",
    "tranche_H3.json": "3c04b610ffc3d0f9170217a0c1d59052383df3e7889b0706dc9261eaede39259",
    "tranche_H4.json": "8dd12f4e7a17275425743922dc4eb8ab091f1719e1808ea6a6c5407565fbcff1",
    "tranche_H5.json": "2b50bf51216c1038d287681ef3c8935eb690cc21633fd98632818f36fcd52c17",
    "tranche_H6.json": "f4872910f1a5ce983e2053b9d6384ced79c263b5c1a10765682f032496613ad0",
    "tranche_H7.json": "ee8e80ebf1e6f0e8a6131fbd7bf612be6f2ee978792566159ecfb1a11dcc9464",
    "tranche_H8.json": "5b81ce14cca2eea1245b2ef51b3507d26a3d8eae385157c83316d4a2d2c539c1",
    "tranche_H9.json": "c9e5cc98c13647d809945312a8e4137ab5dc39731e6a90ca367c4233afdba550",
    "tranche_H10.json": "8596b601b199def60876ee4dc29c6297c0bbe06ebd893c1ac4011bf253e166de",
    "tranche_H11.json": "96f39bdcaf501842d5202f6f24789275a8bac035e54f5c931fb0ce83bb605f88",
    "tranche_H12.json": "5aed124a94f0d398fabbe89d51201fa66af58613f8e89f6870a9d497461027f7",
}

# Une mesure : (nom, mesure). Un test : (identifiant, famille, A, B, sens, énoncé) ; l'écart du test est A − B.
S10 = ("SEUL10", "V_cal")
TESTS = [
    ("P1", "principal", ("K4", "E_cal"), S10, +1,
     "à calcul égal, la recette coupée en quatre réseaux, lus en ensemble, bat le réseau seul"),
    ("P2", "principal", S10, ("K32", "E_cal"), +1,
     "à calcul égal, le réseau seul bat la recette coupée en 32 réseaux, lus en ensemble"),
    ("P3", "principal", ("SOUPE", "V_cal"), S10, +1,
     "à calcul égal et au même coût d'usage, la soupe (la moyenne des poids de quatre enfants d'un même parent) bat le "
     "réseau seul"),
    ("P4", "principal", ("TOILE", "E_cal"), ("K4", "E_cal"), +1,
     "à calcul égal et au même coût d'usage, l'ensemble des quatre instantanés d'un entraînement au taux cyclique bat "
     "quatre réseaux indépendants"),
    ("S1", "secondaire", ("K2", "E_cal"), S10, +1,
     "à calcul égal, la recette coupée en deux réseaux, lus en ensemble, bat le réseau seul"),
    ("S2", "secondaire", ("K8", "E_cal"), S10, +1,
     "à calcul égal, la recette coupée en huit réseaux, lus en ensemble, bat le réseau seul"),
    ("S3", "secondaire", ("K4", "E_cal"), ("K32", "E_cal"), +1,
     "à calcul égal, quatre réseaux battent trente-deux réseaux (la recette coupée, lue en ensemble)"),
    ("S4", "secondaire", ("POPC.TOUS", "E_cal"), ("INDEPC.TOUS", "E_cal"), +1,
     "l'ensemble des 32 réseaux d'une recherche aléatoire aux réglages faits pour le convolutif, sans curation, bat "
     "celui de la recherche aléatoire de la clé 46"),
    ("S5", "secondaire", ("K32", "E_cal"), ("POPC.TOUS", "E_cal"), +1,
     "à 32 réseaux, la recette réglée (taux en cosinus) bat les réglages tirés au hasard (taux constant)"),
    ("S6", "secondaire", ("SWA", "V_cal"), S10, +1,
     "la moyenne des poids du réseau seul sur ses 30 dernières générations (SWA) bat le réseau seul"),
    ("S7", "secondaire", ("EMA", "V_cal"), S10, +1,
     "la moyenne mobile des poids du réseau seul (EMA) bat le réseau seul"),
    ("S8", "secondaire", ("PHOTO", "V_cal"), S10, +1,
     "le réseau seul distillé de l'ensemble K4 (deux fois le calcul) bat le réseau seul"),
    ("S9", "secondaire", ("K4", "E_cal"), ("PHOTO", "V_cal"), +1,
     "l'ensemble K4 bat sa photographie (le réseau seul qui en est distillé)"),
    ("S10", "secondaire", ("SOUPE", "V_cal"), ("MOY_K4", "V_cal"), +1,
     "la soupe d'enfants d'un même parent bat la moyenne des poids de quatre réseaux indépendants (sans ancêtre commun)"),
    ("S11", "secondaire", ("SOUPE.ENFANTS", "E_cal"), ("SOUPE", "V_cal"), +1,
     "l'ensemble des quatre enfants bat leur soupe"),
    ("S12", "secondaire", ("TOILE", "E_cal"), ("TOILE.DERNIER", "V_cal"), +1,
     "l'ensemble des quatre instantanés bat le dernier d'entre eux"),
    ("S13", "secondaire", ("K4", "AMB"), ("TOILE", "AMB"), +1,
     "quatre réseaux indépendants sont plus divers (ambiguïté sur le juge) que les quatre instantanés d'un même "
     "entraînement"),
]
LECTURES = {
    "L1": ("à calcul égal, la meilleure recette coupée en plusieurs réseaux, lus en ensemble, bat le réseau seul "
           "(au moins une de P1, S1, S2 confirmée)", ("P1", "S1", "S2"), any),
    "L2": ("à calcul égal et au même coût d'usage, un seul réseau recueille le multiple (au moins une de P3, S6, S7 "
           "confirmée)", ("P3", "S6", "S7"), any),
}


def lire_front(front, ancre):
    """(V au seuil réglé sur l'ancre, biais retenu, V au meilleur seuil du juge, meilleur biais, V au biais 0)."""
    b = np.array(front["biais"]); V = W_CHAT * np.array(front["Rf"]) + (1 - W_CHAT) * np.array(front["Ao"])
    assert ancre["biais"] == list(front["biais"]), "les grilles du front de l'ancre et du juge diffèrent"
    note = np.array(ancre["note"])
    meilleurs = sorted(np.array(ancre["biais"])[note == note.max()])
    bc = float(meilleurs[(len(meilleurs) - 1) // 2])
    i = int(np.flatnonzero(b == bc)[0])
    i0 = int(np.flatnonzero(b == 0.0)[0])
    return float(V[i]), bc, float(V.max()), float(b[int(V.argmax())]), float(V[i0])


def m_ensemble(e):
    """Un ensemble lu par finales_ensemble (front, ancre, diversite) ou par une règle du choix final."""
    E_cal, eb, E_max, ebm, E0 = lire_front(e["front"], e["ancre"])
    return dict(E_cal=E_cal, eb_cal=eb, E_max=E_max, E_fin=E0, AMB=e["diversite"]["amb"],
                AMB_chat=e["diversite"]["amb_chat"])


def m_reseau(f):
    """Un réseau seul : un front du juge (biais, Rf, Ao) et son front de l'ancre (ancre)."""
    V_cal, b, V_max, bm, V0 = lire_front(f, f["ancre"])
    return dict(V_cal=V_cal, b_cal=b, V_max=V_max, V_fin=V0)


def m_membres(fm):
    vs = [m_reseau(dict(biais=fm["biais"], Rf=r["Rf"], Ao=r["Ao"], ancre=r["ancre"])) for r in fm["reseaux"]]
    return dict(V_cal=float(np.mean([v["V_cal"] for v in vs])), V_fin=float(np.mean([v["V_fin"] for v in vs])),
                V_max=float(np.mean([v["V_max"] for v in vs])), V_cal_max=float(max(v["V_cal"] for v in vs)),
                V_cal_tous=[v["V_cal"] for v in vs])


def lire_tranches(motif=None, empreintes=None, dossier=None):
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
                a = json.dumps({x: y for x, y in D[k].items() if x != "duree"}, sort_keys=True)
                b = json.dumps({x: y for x, y in v.items() if x != "duree"}, sort_keys=True)
                assert a == b, f"course en double, différente : {k}"
            D[k] = v
    return D


def toutes_mesures(D, R46):
    """M[nom][graine] = dict des mesures."""
    M = {}

    def met(nom, s, d):
        M.setdefault(nom, {})[s] = d
    for s in GRAINES:
        for b in ("K2", "K8", "K32", "K4P"):
            r = D.get(f"cnn_{b}_s{s}")
            if r is None:
                continue
            n = "K4" if b == "K4P" else b
            met(n, s, m_ensemble(r["fin_ens"]))
            met(n + ".MEMBRES", s, m_membres(r["fin_membres"]))
            met("MOY_" + n, s, m_reseau(r["fin_moyenne"]))
            if b == "K4P" and "photo" in r:
                met("PHOTO", s, m_reseau(r["photo"]["fin_livre"]))
        r = D.get(f"cnn_POPC_s{s}")
        if r is not None:
            for nom, g in r["fin_choix"]["regles"].items():
                met("POPC." + nom, s, m_ensemble(g))
            met("POPC.LIVRE", s, m_reseau(r["fin_livre"]))
        r = D.get(f"cnn_VEILLEUR_s{s}")
        if r is not None:
            met("SEUL10", s, m_reseau(r["fin_livre"]))
            met("SWA", s, m_reseau(r["fin_swa"]))
            met("EMA", s, m_reseau(r["fin_ema"]))
        r = D.get(f"cnn_TOILE_s{s}")
        if r is not None:
            met("TOILE", s, m_ensemble(r["fin_ens"]))
            met("TOILE.DERNIER", s, m_reseau(r["fin_livre"]))
            met("TOILE.MEMBRES", s, m_membres(r["fin_membres"]))
            met("MOY_TOILE", s, m_reseau(r["fin_moyenne"]))
        r = D.get(f"cnn_SOUPE_s{s}")
        if r is not None:
            met("SOUPE", s, m_reseau(r["fin_soupe"]))
            met("GOURMANDE", s, m_reseau(r["fin_gourmande"]))
            met("SOUPE.ENFANTS", s, m_ensemble(r["fin_ens"]))
            met("SOUPE.MEMBRES", s, m_membres(r["fin_membres"]))
        r = R46.get(f"cnn_INDEPC_s{s}")
        if r is not None:
            for nom, g in r["fin_choix"]["regles"].items():
                met("INDEPC." + nom, s, m_ensemble(g))
        for b in ("SEUL10", "SEUL", "COURT", "GRAND"):
            r = R46.get(f"cnn_{b}_s{s}")
            if r is not None:
                met(b + "_46", s, m_reseau(r["fin_livre"]))
    return M


def identites(D, R46):
    """Le réseau du VEILLEUR doit refaire au bit près la course SEUL10 de la clé 46 (trajet et mesures finales), et le
    parent de la SOUPE sa première moitié (les générations 0 à 59 de son trajet)."""
    out = dict(veilleur=dict(identiques=0, differents=[]), soupe_parent=dict(identiques=0, differents=[]))
    for s in GRAINES:
        a, b, c = D.get(f"cnn_VEILLEUR_s{s}"), R46.get(f"cnn_SEUL10_s{s}"), D.get(f"cnn_SOUPE_s{s}")
        if a is not None and b is not None:
            same = (json.dumps(a["tr"], sort_keys=True) == json.dumps(b["tr"], sort_keys=True)
                    and json.dumps(a["juge"], sort_keys=True) == json.dumps(b["juge"], sort_keys=True)
                    and json.dumps(a["fin_livre"], sort_keys=True) == json.dumps(b["fin_livre"], sort_keys=True))
            if same:
                out["veilleur"]["identiques"] += 1
            else:
                out["veilleur"]["differents"].append(s)
        if c is not None and b is not None:
            gm = c["calcul"]["mi_course"]
            ta = [x for x in c["tr"] if x["g"] < gm]; tb = [x for x in b["tr"] if x["g"] < gm]
            ja = [x for x in c["juge"] if x["g"] < gm]; jb = [x for x in b["juge"] if x["g"] < gm]
            same = (json.dumps(ta, sort_keys=True) == json.dumps(tb, sort_keys=True)
                    and json.dumps(ja, sort_keys=True) == json.dumps(jb, sort_keys=True))
            if same:
                out["soupe_parent"]["identiques"] += 1
            else:
                out["soupe_parent"]["differents"].append(s)
    return out


def serie(M, X):
    nom, m = X
    return {s: v[m] for s, v in M.get(nom, {}).items() if m in v}


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


def appliquer(M):
    res = []
    for tid, fam, A, B, sens, texte in TESTS:
        a, b = serie(M, A), serie(M, B)
        g = sorted(set(a) & set(b))
        d = {s: a[s] - b[s] for s in g}
        manque = len(GRAINES) - len(g)
        e = dict(id=tid, famille=fam, A=".".join(A), B=".".join(B), enonce=texte, manquantes=manque)
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
    D = lire_tranches(os.path.join(arg("--resultats", os.path.join(ICI, "resultats")), "tranche_J*.json"))
    R46 = lire_tranches(None, EMPREINTES46, arg("--cle46", os.path.join(ICI, "..", "cle46", "resultats")))
    sys.path.insert(0, ICI)
    import plan47
    attendues = [plan47.nom(t) for v in plan47.tranches().values() for t in v]
    absentes = [k for k in attendues if k not in D]
    en_trop = sorted(set(D) - set(attendues))
    M = toutes_mesures(D, R46)
    res, lectures = appliquer(M)
    ident = identites(D, R46)
    json.dump(dict(courses=len(D), absentes=absentes, en_trop=en_trop, identites=ident, tests=res, lectures=lectures),
              open(arg("--sortie", os.path.join(ICI, "verdicts47.json")), "w"), ensure_ascii=False, indent=1)
    print(f"courses du plan : {len(attendues) - len(absentes)}/{len(attendues)} ; absentes : {absentes or 'aucune'} ; "
          f"en trop : {en_trop or 'aucune'}")
    print(f"identités : VEILLEUR = SEUL10 de la clé 46 sur {ident['veilleur']['identiques']} graines "
          f"(différentes : {ident['veilleur']['differents'] or 'aucune'}) ; parent de la SOUPE = première moitié de "
          f"SEUL10 sur {ident['soupe_parent']['identiques']} (différentes : {ident['soupe_parent']['differents'] or 'aucune'})")
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        print(f"{e['id']:<4} {e['verdict']:<11} écart {100 * e['ecart']:+7.3f}  t = {e['t']:+6.2f}  p = {e['p']:.2g}  "
              f"(Holm sur 17 : {e['p_holm_tous']:.2g})  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else ('non' if v['tenue'] is False else 'incomplète')}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
