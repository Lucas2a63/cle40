"""Le verdict de la clé 48, selon la règle écrite dans PREENREGISTREMENT_48.md :
python3 verdict48.py --cle47 DOSSIER [--resultats DOSSIER] [--sortie FICHIER]   (--cle47 est obligatoire)

Lit resultats/tranche_J*.json (les 264 courses du plan : onze bras sur le réseau convolutif, en 32×32 ; partie A,
graines 124 à 147 ; partie B, graines 100 à 123) et, comme références de la partie B, graine par graine, les courses
de la clé 47 réunies (les 24 fichiers tranche_J*.json de son dossier resultats, publiés dans son archive ; leurs
empreintes sont vérifiées ici) : le réseau seul (SEUL10, le réseau du VEILLEUR), K2, K4, K8 et K32. Calcule les mesures
pré-enregistrées, applique les 19 tests (4 principaux, corrigés ensemble par la méthode de Holm ; 15 secondaires, dont
2 d'équivalence) et les trois lectures, vérifie que toutes les courses de la clé 48 d'une même graine ont la même
empreinte du monde, et écrit verdicts48.json.

Mesures — le seuil réglé sur l'ancre est celui des clés 42 à 47 : la médiane basse des biais du chat qui maximisent la
note de l'ancre (sur les ancres réunies des dix dernières générations) ; la vérité ×30 du juge
V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87 calculé à partir des proportions du monde :
  pour un ensemble (K2_H, le maître de PHOTO2 ; K2_Q, K4_Q, K2_S, K4_S ; l'ensemble des instantanés de TOILE4 et
    TOILE8 ; K2, K4 de la clé 47) : E_cal, sa vérité au seuil réglé sur son propre front de l'ancre ; E_fin au biais 0 ;
    E_max au meilleur biais du juge (un diagnostic) ; AMB, l'ambiguïté sur le juge ;
  pour un réseau seul (SEUL10, SEUL_H, SEUL_Q, SEUL_S ; PHOTO2, l'élève ; MOY_TOILE4 et MOY_TOILE8, la moyenne des poids
    de tous les instantanés ; MOY_TOILE4_FIN et MOY_TOILE8_FIN, celle de la seconde moitié ; TOILE4.DERNIER et
    TOILE8.DERNIER, le dernier instantané ; les moyennes de poids sans ancêtre commun MOY_K2_H, MOY_K2_Q, … ; SEUL10_47) :
    V_cal, au seuil réglé sur son propre front de l'ancre ; V_fin ; V_max ; LOG, l'échelle de ses logits sur le juge
    (fin_confiance : l'écart-type des dix logits d'une image, moyen sur les images) ;
  pour les réseaux d'un ensemble pris seuls : V_cal de chacun, et leur moyenne (MEMBRES) ;
  dérivées, par graine de la partie B : GAIN (ensemble moins réseau seul au même calcul : GAIN.K2_47 = K2 − SEUL10 de la
    clé 47, GAIN.K2_Q, GAIN.K2_S), GMEM (ensemble moins la moyenne de ses réseaux pris seuls : GMEM.K2_47, GMEM.K2_S),
    PRIX (le réseau seul moins les réseaux de K2 au même calcul total, chacun nourri de la moitié : PRIX.Q, PRIX.S).
Écart d'un test, par graine : A − B, sur les 24 graines de sa partie. Règle : Student apparié ; p unilatéral exact.
  principaux : confirmée si Holm (0,025 unilatéral, sur les 4) la retient ; démentie si le p contraire < 0,025 / 4 ;
               indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  équivalence (marge ±0,5 point, deux tests unilatéraux) : confirmée si les deux p < 0,025 ; démentie si l'écart est
               établi hors de la marge (p < 0,025 pour « écart > +0,5 » ou pour « écart < −0,5 ») ; indécidable sinon.
  Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens. Une graine dont l'écart n'est pas un nombre fini compte
  comme manquante. Plus de deux graines manquantes : « incomplet » ; un test principal incomplet compte p = 1 dans la
  correction de Holm, qui porte toujours sur les quatre.
Une lecture « au moins une » tient si l'un de ses tests est confirmé, ne tient pas si aucun ne l'est et qu'aucun n'est
incomplet, et reste incomplète sinon ; une lecture « toutes » ne tient pas si l'un de ses tests complets n'est pas
confirmé, tient si tous sont confirmés, et reste incomplète sinon.
Lectures (sans test supplémentaire) :
  L1 « à calcul égal et au même coût d'usage, un seul réseau recueille une part du multiple » : au moins une de P1,
     P2, S1 confirmée ;
  L2 « le découpage gagne encore quand le calcul manque » : P4 confirmée ;
  L3 « la moitié du calcul suffisait au réseau seul, et le découpage gagne moins quand le calcul manque » : S9 et S12
     confirmées.
"""
import glob, hashlib, json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRAINES_A = list(range(124, 148))
GRAINES_B = list(range(100, 124))
ALPHA = 0.025
MARGE = 0.005                                    # la marge des tests d'équivalence : ±0,5 point de vérité
P_MONDE = np.full(10, (1 - 0.03) / 9); P_MONDE[3] = 0.03
COUT = np.ones(10); COUT[3] = 30.0
W_CHAT = float((P_MONDE * COUT / (P_MONDE * COUT).sum())[3])

# Les courses de la clé 47 utilisées comme références (empreintes SHA-256 des fichiers réunis, cle47/resultats/).
EMPREINTES47 = {
    "tranche_J1.json": "ab9f4d4b5033b0e2311e3fe0d3a6f6d45105f9b724b89cc752afccebdf208818",
    "tranche_J2.json": "32ae409ea8661625af9acf41a8e673870b49afcf063f78920813f0df8c2b28af",
    "tranche_J3.json": "291f3221daefb8421681f899164e54612f35b005e0833405a5d9167cedd5814a",
    "tranche_J4.json": "0ec2e6ae86a90cc58566e0cc6ec2ab28f7c64427787f9584cf57444fdbd42495",
    "tranche_J5.json": "af0524b9daa32a0125fa9673ce549ab27e531e769df40530a9f05dba0861ff6a",
    "tranche_J6.json": "2286a04a0a4bdd8ef46f6d66aca2a1b042a4fb746d4f97b2698924cbd5da4313",
    "tranche_J7.json": "fb9c97a593d63289470efda8540af9810d217f47d748a015f2dbef64b55a601e",
    "tranche_J8.json": "d6d8b5a94ba1544da44e171255c4d4a8bf53b234ba2a0e83c09696e887bd0f11",
    "tranche_J9.json": "7905261504af79990e58599eb6b6fc5f2190cb2fb1da7bdcccd453916beaf1f7",
    "tranche_J10.json": "f8f75222ff0ea81764aefef39af1aa631fc12cf2e1137a183409d53e7d1d4f67",
    "tranche_J11.json": "5b62f36ebcb8e7b0e32bd5d676d1852fa551df32ebc0026917275216b46513e4",
    "tranche_J12.json": "d240890c005517bd01d83a4be1b8694f7e037e7c1071897d5b8e3ca78fbbe736",
    "tranche_J13.json": "e90b9cd8989b5ee0fe4799a72a79d3a32eff7e9e16643fb013ea8eec86ca62ca",
    "tranche_J14.json": "841cb39eebf081c1b97f9434ea96d7eba520e841421108a46b7d69e8ee1a407b",
    "tranche_J15.json": "a5caaebb203ca29e7636abfc704c08477f0125d6adc9e8b10183fb47f82e2811",
    "tranche_J16.json": "51344689a9d0d8e954c76d2adc0b332a81c6312beeac8079509697966b095e03",
    "tranche_J17.json": "12da947e85ccb145ca5aa6d0e0b8a4b471738f60df1046f9684da8ca4bf91221",
    "tranche_J18.json": "e9b97469cfc57cc599cb5b24e2dc26d4e8087902a6e423949a1eb1645d27aebb",
    "tranche_J19.json": "c888a84f918d38b3bbb49882d7c2a2000d9a4269ffcf929bc681fdc209e9a2f7",
    "tranche_J20.json": "1b7bb4cc52bb28300cf332697178aa2ab001688b172b79f6a6832a6799751133",
    "tranche_J21.json": "cb9534143d042f1f0ba1f553830da100b15aa522c47b627e085cfaafce6d99f0",
    "tranche_J22.json": "7d8077f2ebc6c882a076c188b5e56d6775350d49d5e7c89dfb0aba2a82e45635",
    "tranche_J23.json": "4f6830601d25ca11597ccbe9de8be221843788ae62499837508e28e00c5f61f8",
    "tranche_J24.json": "43eebe13d0ae5ad41155d26d72ac0edeace51e3732ec680cc56deb8fd894dd72",
}

# Un test : (identifiant, famille, A, B, sens ou marge, unité, partie, énoncé) ; l'écart du test est A − B.
S10 = ("SEUL10", "V_cal")
TESTS = [
    ("P1", "principal", ("MOY_TOILE4", "V_cal"), S10, +1, "points", "A",
     "à calcul égal et au même coût d'usage, la moyenne des poids des quatre instantanés d'un entraînement au taux "
     "cyclique bat le réseau seul"),
    ("P2", "principal", ("PHOTO2", "V_cal"), S10, +1, "points", "A",
     "à calcul total égal (le maître et l'élève se partagent le calcul du réseau seul) et au même coût d'usage, la "
     "distillation bat le réseau seul"),
    ("P3", "principal", ("K2_Q", "E_cal"), ("SEUL_Q", "V_cal"), +1, "points", "B",
     "au quart du calcul, la recette coupée en deux réseaux, lus en ensemble, bat le réseau seul"),
    ("P4", "principal", ("K2_S", "E_cal"), ("SEUL_S", "V_cal"), +1, "points", "B",
     "au seizième du calcul, la recette coupée en deux réseaux, lus en ensemble, bat le réseau seul"),
    ("S1", "secondaire", ("MOY_TOILE8", "V_cal"), S10, +1, "points", "A",
     "la moyenne des poids des huit instantanés d'un entraînement à huit cycles bat le réseau seul"),
    ("S2", "equivalence", ("MOY_TOILE4", "V_cal"), ("TOILE4", "E_cal"), MARGE, "points", "A",
     "la moyenne des poids des quatre instantanés vaut leur ensemble, à 0,5 point près"),
    ("S3", "secondaire", ("TOILE4", "E_cal"), S10, +1, "points", "A",
     "l'ensemble des quatre instantanés (quatre réseaux à l'usage) bat le réseau seul"),
    ("S4", "secondaire", ("SEUL10", "V_fin"), ("MOY_TOILE4", "V_fin"), +1, "points", "A",
     "au seuil par défaut (biais 0), le réseau seul bat la moyenne des poids des quatre instantanés"),
    ("S5", "secondaire", ("SEUL10", "LOG"), ("MOY_TOILE4", "LOG"), +1, "logits", "A",
     "les logits de la moyenne des poids des quatre instantanés sont plus petits que ceux du réseau seul"),
    ("S6", "secondaire", ("PHOTO2", "V_cal"), ("SEUL_H", "V_cal"), +1, "points", "A",
     "l'élève bat son jumeau sans maître (mêmes poids de départ, mêmes paquets, même calcul)"),
    ("S7", "secondaire", ("K2_H", "E_cal"), S10, +1, "points", "A",
     "deux réseaux qui se partagent la moitié du calcul, lus en ensemble, battent le réseau seul nourri de tout le "
     "calcul"),
    ("S8", "secondaire", ("K2_H", "E_cal"), ("PHOTO2", "V_cal"), +1, "points", "A",
     "le maître (deux réseaux, la moitié du calcul) bat sa photographie"),
    ("S9", "equivalence", ("SEUL_H", "V_cal"), S10, MARGE, "points", "A",
     "le réseau seul nourri de la moitié du calcul vaut le réseau seul nourri de tout le calcul, à 0,5 point près"),
    ("S10", "secondaire", ("K4_Q", "E_cal"), ("SEUL_Q", "V_cal"), +1, "points", "B",
     "au quart du calcul, la recette coupée en quatre réseaux, lus en ensemble, bat le réseau seul"),
    ("S11", "secondaire", ("SEUL_S", "V_cal"), ("K4_S", "E_cal"), +1, "points", "B",
     "au seizième du calcul, le réseau seul bat la recette coupée en quatre réseaux"),
    ("S12", "secondaire", ("GAIN.K2_47", "d"), ("GAIN.K2_S", "d"), +1, "points", "B",
     "ce que la coupe en deux gagne sur le réseau seul est plus grand à tout le calcul (clé 47) qu'au seizième"),
    ("S13", "secondaire", ("GMEM.K2_47", "d"), ("GMEM.K2_S", "d"), +1, "points", "B",
     "ce que l'ensemble de deux réseaux gagne sur ses réseaux pris seuls est plus grand à tout le calcul (clé 47) "
     "qu'au seizième"),
    ("S14", "secondaire", ("PRIX.S", "d"), ("PRIX.Q", "d"), +1, "points", "B",
     "diviser par deux le calcul d'un réseau seul lui coûte plus au seizième du calcul qu'au quart"),
    ("S15", "secondaire", ("K2_47", "AMB"), ("K2_S", "AMB"), +1, "Brier", "B",
     "deux réseaux nourris chacun de la moitié du calcul (clé 47) sont plus divers (ambiguïté sur le juge) que deux "
     "réseaux nourris chacun du trente-deuxième"),
]
LECTURES = {
    "L1": ("à calcul égal et au même coût d'usage, un seul réseau recueille une part du multiple (au moins une de P1, "
           "P2, S1 confirmée)", ("P1", "P2", "S1"), any),
    "L2": ("le découpage gagne encore quand le calcul manque (P4 confirmée)", ("P4",), all),
    "L3": ("la moitié du calcul suffisait au réseau seul, et le découpage gagne moins quand le calcul manque (S9 et S12 "
           "confirmées)", ("S9", "S12"), all),
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
    """Un ensemble lu par finales_ensemble (front, ancre, diversite)."""
    E_cal, eb, E_max, ebm, E0 = lire_front(e["front"], e["ancre"])
    return dict(E_cal=E_cal, eb_cal=eb, E_max=E_max, E_fin=E0, AMB=e["diversite"]["amb"],
                AMB_chat=e["diversite"]["amb_chat"])


def m_reseau(f, conf=None):
    """Un réseau seul : un front du juge (biais, Rf, Ao) et son front de l'ancre (ancre) ; conf : sa confiance."""
    V_cal, b, V_max, bm, V0 = lire_front(f, f["ancre"])
    d = dict(V_cal=V_cal, b_cal=b, V_max=V_max, V_fin=V0)
    if conf is not None:
        d.update(LOG=conf["ecart_logits"], CONF=conf["conf"], NLL=conf["nll"], PCHAT=conf["p_chat_chats"],
                 ENT=conf["entropie"])
    return d


def m_membres(fm):
    vs = [m_reseau(dict(biais=fm["biais"], Rf=r["Rf"], Ao=r["Ao"], ancre=r["ancre"])) for r in fm["reseaux"]]
    return dict(V_cal=float(np.mean([v["V_cal"] for v in vs])), V_fin=float(np.mean([v["V_fin"] for v in vs])),
                V_max=float(np.mean([v["V_max"] for v in vs])), V_cal_max=float(max(v["V_cal"] for v in vs)),
                V_cal_tous=[v["V_cal"] for v in vs], b_cal_tous=[v["b_cal"] for v in vs])


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


def toutes_mesures(D, R47):
    """M[nom][graine] = dict des mesures."""
    M = {}

    def met(nom, s, d):
        M.setdefault(nom, {})[s] = d

    def coupe(nom, s, r):
        met(nom, s, m_ensemble(r["fin_ens"]))
        met(nom + ".MEMBRES", s, m_membres(r["fin_membres"]))
        met("MOY_" + nom, s, m_reseau(r["fin_moyenne"], r.get("fin_confiance", {}).get("moyenne")))
    for s in GRAINES_A:
        for b in ("SEUL10", "SEUL_H"):
            r = D.get(f"cnn_{b}_s{s}")
            if r is not None:
                met(b, s, m_reseau(r["fin_livre"], r["fin_confiance"]["livre"]))
        r = D.get(f"cnn_PHOTO2_s{s}")
        if r is not None:
            coupe("K2_H", s, r)
            met("PHOTO2", s, m_reseau(r["photo"]["fin_livre"], r["photo"]["fin_confiance"]["livre"]))
        for c in (4, 8):
            r = D.get(f"cnn_TOILE{c}_s{s}")
            if r is not None:
                fc = r["fin_confiance"]
                met(f"TOILE{c}", s, m_ensemble(r["fin_ens"]))
                met(f"TOILE{c}.DERNIER", s, m_reseau(r["fin_livre"], fc["livre"]))
                met(f"TOILE{c}.MEMBRES", s, m_membres(r["fin_membres"]))
                met(f"MOY_TOILE{c}", s, m_reseau(r["fin_moyenne"], fc["moyenne"]))
                met(f"MOY_TOILE{c}_FIN", s, m_reseau(r["fin_moyenne_fin"], fc["moyenne_fin"]))
    for s in GRAINES_B:
        for b in ("SEUL_Q", "SEUL_S"):
            r = D.get(f"cnn_{b}_s{s}")
            if r is not None:
                met(b, s, m_reseau(r["fin_livre"], r["fin_confiance"]["livre"]))
        for b in ("K2_Q", "K4_Q", "K2_S", "K4_S"):
            r = D.get(f"cnn_{b}_s{s}")
            if r is not None:
                coupe(b, s, r)
        r = R47.get(f"cnn_VEILLEUR_s{s}")
        if r is not None:
            met("SEUL10_47", s, m_reseau(r["fin_livre"]))
        for b, n in (("K2", "K2_47"), ("K4P", "K4_47"), ("K8", "K8_47"), ("K32", "K32_47")):
            r = R47.get(f"cnn_{b}_s{s}")
            if r is not None:
                coupe(n, s, r)
    # les mesures dérivées de la partie B, graine par graine
    for s in GRAINES_B:
        def v(nom, m):
            return M.get(nom, {}).get(s, {}).get(m)
        for nom, a, b in (("GAIN.K2_47", ("K2_47", "E_cal"), ("SEUL10_47", "V_cal")),
                          ("GAIN.K2_Q", ("K2_Q", "E_cal"), ("SEUL_Q", "V_cal")),
                          ("GAIN.K2_S", ("K2_S", "E_cal"), ("SEUL_S", "V_cal")),
                          ("GMEM.K2_47", ("K2_47", "E_cal"), ("K2_47.MEMBRES", "V_cal")),
                          ("GMEM.K2_Q", ("K2_Q", "E_cal"), ("K2_Q.MEMBRES", "V_cal")),
                          ("GMEM.K2_S", ("K2_S", "E_cal"), ("K2_S.MEMBRES", "V_cal")),
                          ("PRIX.Q", ("SEUL_Q", "V_cal"), ("K2_Q.MEMBRES", "V_cal")),
                          ("PRIX.S", ("SEUL_S", "V_cal"), ("K2_S.MEMBRES", "V_cal"))):
            x, y = v(*a), v(*b)
            if x is not None and y is not None:
                met(nom, s, dict(d=x - y))
    return M


def meme_monde(D, R47=None):
    """Toutes les courses de la clé 48 d'une même graine, l'élève de PHOTO2 compris, ont la même empreinte du monde
    (« monde » : l'état du générateur du monde, la réserve déjà tirée, le tampon et les ancres à la fin de
    l'entraînement). Les courses de la clé 47 n'ont pas d'empreinte : pour la partie B, le même monde est garanti par le
    code (même moteur, mêmes graines), sans donnée pour le vérifier ; R47 n'est pas lu."""
    out = dict(identiques=0, differentes=[], sans_empreinte=[])
    for s in GRAINES_A + GRAINES_B:
        vus = []
        for k, r in D.items():
            if k.endswith(f"_s{s}"):
                for x in (r, r.get("photo")):
                    if x is None:
                        continue
                    if "monde" in x:
                        vus.append(x["monde"])
                    else:
                        out["sans_empreinte"].append(k)
        if not vus:
            continue
        if len(set(vus)) == 1:
            out["identiques"] += 1
        else:
            out["differentes"].append(s)
    return out


def serie(M, X):
    nom, m = X
    return {s: v[m] for s, v in M.get(nom, {}).items() if m in v}


def student(d, decalage=0.0):
    """Student sur d − decalage : (n, moyenne de d, écart-type, t, p « > », p « < »)."""
    d = np.asarray(d, float)
    assert np.isfinite(d).all(), "un écart n'est pas un nombre fini"
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    x = d.mean() - decalage
    if se > 0:
        t = x / se
    else:
        t = 0.0 if x == 0 else math.copysign(math.inf, x)
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
    for tid, fam, A, B, sens, unite, partie, texte in TESTS:
        graines = GRAINES_A if partie == "A" else GRAINES_B
        a, b = serie(M, A), serie(M, B)
        g = sorted(set(a) & set(b) & set(graines))
        d = {s: a[s] - b[s] for s in g}
        non_finies = sorted(s for s, x in d.items() if not math.isfinite(x))
        d = {s: x for s, x in d.items() if math.isfinite(x)}
        manque = len(graines) - len(d)
        e = dict(id=tid, famille=fam, A=".".join(A), B=".".join(B), unite=unite, partie=partie, enonce=texte,
                 manquantes=manque, non_finies=non_finies)
        if manque > 2 or len(d) < 3:
            e.update(verdict="incomplet"); res.append(e); continue
        n, moy, sd, t, p_haut, p_bas = student(list(d.values()))
        se = sd / math.sqrt(n)
        q = float(stats.t.ppf(0.975, n - 1))
        e.update(n=n, ecart=moy, sd=sd, t=t, ic95=[moy - q * se, moy + q * se], gagne=int(sum(x > 0 for x in d.values())),
                 perd=int(sum(x < 0 for x in d.values())))
        if fam == "equivalence":
            p_sup = student(list(d.values()), -sens)[4]       # H0 : écart ≤ −marge
            p_inf = student(list(d.values()), +sens)[5]       # H0 : écart ≥ +marge
            hors_haut = student(list(d.values()), +sens)[4]   # « écart > +marge » établi
            hors_bas = student(list(d.values()), -sens)[5]    # « écart < −marge » établi
            e.update(marge=sens, p=max(p_sup, p_inf), p_sup=p_sup, p_inf=p_inf, p_hors_haut=hors_haut,
                     p_hors_bas=hors_bas, p_contraire=min(hors_haut, hors_bas))
        else:
            p, pc = (p_haut, p_bas) if sens > 0 else (p_bas, p_haut)
            e.update(p=p, p_contraire=pc)
        res.append(e)
    prin = [e for e in res if e["famille"] == "principal"]          # les quatre, incomplets compris (p = 1)
    gardes = holm([e.get("p", 1.0) for e in prin], ALPHA)
    for i, e in enumerate(prin):
        if "p" in e:
            e["verdict"] = ("confirmée" if i in gardes else
                            ("démentie" if e["p_contraire"] < ALPHA / len(prin) else "indécidable"))
    for e in res:
        if e["famille"] in ("secondaire", "equivalence") and "p" in e:
            e["verdict"] = "confirmée" if e["p"] < ALPHA else ("démentie" if e["p_contraire"] < ALPHA else "indécidable")
    avec_p = [e for e in res if "p" in e]
    for e, a in zip(avec_p, holm_ajuste([e["p"] for e in avec_p])):
        e["p_holm_tous"] = a
    par_id = {e["id"]: e for e in res}
    lectures = {}
    for lid, (texte, ids, f) in LECTURES.items():
        vs = [par_id[i]["verdict"] for i in ids]
        if f is any:
            tenue = True if "confirmée" in vs else (None if "incomplet" in vs else False)
        else:
            tenue = False if any(v not in ("confirmée", "incomplet") for v in vs) else (None if "incomplet" in vs
                                                                                       else True)
        lectures[lid] = dict(enonce=texte, tests={i: v for i, v in zip(ids, vs)}, tenue=tenue)
    return res, lectures


def arg(nom, defaut):
    return sys.argv[sys.argv.index(nom) + 1] if nom in sys.argv else defaut


def echelle(e):
    return 100.0 if e["unite"] == "points" else 1.0


if __name__ == "__main__":
    D = lire_tranches(os.path.join(arg("--resultats", os.path.join(ICI, "resultats")), "tranche_J*.json"))
    assert "--cle47" in sys.argv, "--cle47 DOSSIER est obligatoire (les 24 fichiers réunis de la clé 47)"
    R47 = lire_tranches(None, EMPREINTES47, arg("--cle47", None))
    sys.path.insert(0, ICI)
    import plan48
    attendues = [plan48.nom(t) for v in plan48.tranches().values() for t in v]
    absentes = [k for k in attendues if k not in D]
    en_trop = sorted(set(D) - set(attendues))
    M = toutes_mesures(D, R47)
    res, lectures = appliquer(M)
    monde = meme_monde(D)
    json.dump(dict(courses=len(D), absentes=absentes, en_trop=en_trop, meme_monde=monde, tests=res, lectures=lectures),
              open(arg("--sortie", os.path.join(ICI, "verdicts48.json")), "w"), ensure_ascii=False, indent=1)
    print(f"courses du plan : {len(attendues) - len(absentes)}/{len(attendues)} ; absentes : {absentes or 'aucune'} ; "
          f"en trop : {en_trop or 'aucune'}")
    print(f"même empreinte du monde pour toutes les courses d'une graine : {monde['identiques']} graines "
          f"(différentes : {monde['differentes'] or 'aucune'} ; sans empreinte : {monde['sans_empreinte'] or 'aucune'})")
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        k = echelle(e)
        print(f"{e['id']:<4} {e['verdict']:<11} écart {k * e['ecart']:+8.3f} {e['unite']:<6} "
              f"[{k * e['ic95'][0]:+.3f} ; {k * e['ic95'][1]:+.3f}]  t = {e['t']:+6.2f}  p = {e['p']:.2g}  "
              f"(Holm, tests complets : {e['p_holm_tous']:.2g})  {e['enonce']}")
    for lid, v in lectures.items():
        print(f"lecture {lid} ({v['enonce']}) : {'oui' if v['tenue'] else ('non' if v['tenue'] is False else 'incomplète')}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
