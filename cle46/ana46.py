"""Analyse de la clé 46 (descriptive, et comparaisons sans prédiction, signalées comme telles) :
python3 ana46.py [--cle43 DOSSIER] [--cle45 DOSSIER] → ana46.json

1. Par bras du convolutif (et, pour comparaison, pour le perceptron des clés 43 et 45 sur les mêmes graines) : la
   vérité au seuil réglé sur l'ancre, au seuil par défaut et au meilleur seuil du juge ; le biais retenu ; le rappel du
   chat et la précision ordinaire au seuil réglé ; la vérité à rappel du chat égal (0,85, 0,90, 0,92) ; pour les
   règles du choix final, le nombre de réseaux distincts, la surestimation au biais 0 (H0) et la diversité de
   l'ensemble (ambiguïté, désaccord, Brier moyen et de l'ensemble) ; les durées.
2. Comparaisons sans prédiction, appariées par graine : chaque bras et chaque règle du convolutif contre l'ensemble des
   32 (TOUS), et contre le réseau seul pesé à la 10e génération (SEUL10), à seuil égal, au seuil par défaut et au
   meilleur seuil ; à rappel du chat égal ; le convolutif contre le perceptron, bras par bras (l'ensemble, son choix,
   le réseau livré, le grand réseau, le réseau seul) ; ce que le réglage du seuil rapporte à chaque bras (E_cal −
   E_fin) ; la diversité de l'ensemble convolutif contre celle de l'ensemble des perceptrons.
3. Ce que choisit la sélection gloutonne sur le convolutif : l'étape retenue (et combien de courses à la borne de 64),
   le poids du réseau le plus lourd, les réglages et la vérité propre des réseaux retenus et des autres, la présence du
   réseau livré ; la corrélation, sur les 32 réseaux d'une course, entre le critère pris seul et la vérité propre ;
   l'élagage (combien de réseaux il retire).
4. La qualité des 32 réseaux convolutifs de la recherche aléatoire : leur vérité propre au seuil par défaut (moyenne,
   écart-type, minimum et maximum dans une course), contre celle des 32 perceptrons.
Écrit avant toute course du plan : n'a rien affiché des données de la clé 46.
"""
import json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
from verdict46 import (GRAINES, REGLES, RECETTES, W_CHAT, lire_front, mesures_regle, mesures_reseau, charger,
                       toutes_mesures, arg)


def t_apparie(d):
    d = np.asarray([x for x in d if x is not None], float)
    if len(d) < 3:
        return None
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("nan")
    q = stats.t.ppf(0.975, len(d) - 1)
    return dict(n=len(d), ecart=float(d.mean()), t=float(t), ic95=[float(d.mean() - q * se), float(d.mean() + q * se)],
                gagne=int((d > 0).sum()), perd=int((d < 0).sum()))


def ao_a(front, cible):
    Rf, Ao = np.array(front["Rf"]), np.array(front["Ao"])
    o = np.argsort(Rf, kind="stable")
    return float(np.interp(cible, Rf[o], Ao[o])) if Rf.min() <= cible <= Rf.max() else None


def v_a(front, cible):
    """La vérité à rappel du chat égal à cible : w·cible + (1 − w)·Ao interpolée."""
    a = ao_a(front, cible)
    return None if a is None else W_CHAT * cible + (1 - W_CHAT) * a


def front_et_ancre(course, regle=None):
    if regle is None:
        f = course["fin_livre"]
        return f, f["ancre"]
    g = course["fin_choix"]["regles"][regle]
    return g["front"], g["ancre"]


def rappel_cal(front, ancre):
    V, bc, *_ = lire_front(front, ancre)
    i = front["biais"].index(bc)
    return front["Rf"][i], front["Ao"][i]


def moy(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


if __name__ == "__main__":
    D, R45, R43 = charger(arg("--cle43", os.path.join(ICI, "..", "cle43", "resultats")),
                          arg("--cle45", os.path.join(ICI, "..", "cle45", "resultats")))
    M = toutes_mesures(D, R45, R43)
    G = [s for s in GRAINES if all(s in M.get(("cnn", b), {}) for b in ("TOUS",) + RECETTES)]
    # les courses de chaque objet mesuré : (arch, nom) -> {graine: (course, règle ou None)}
    OBJ = {}
    for s in GRAINES:
        if f"cnn_INDEPC_s{s}" in D:
            for r in REGLES:
                OBJ.setdefault(("cnn", r), {})[s] = (D[f"cnn_INDEPC_s{s}"], r)
            OBJ.setdefault(("cnn", "LIVRE"), {})[s] = (D[f"cnn_INDEPC_s{s}"], None)
        for b in RECETTES:
            if f"cnn_{b}_s{s}" in D:
                OBJ.setdefault(("cnn", b), {})[s] = (D[f"cnn_{b}_s{s}"], None)
        if f"mlp_INDEPC_s{s}" in R45:
            for r in REGLES:
                OBJ.setdefault(("mlp", r), {})[s] = (R45[f"mlp_INDEPC_s{s}"], r)
            OBJ.setdefault(("mlp", "LIVRE"), {})[s] = (R45[f"mlp_INDEPC_s{s}"], None)
        for b in ("COURT2", "GRAND", "SEUL"):
            if f"mlp_{b}_s{s}" in R43:
                OBJ.setdefault(("mlp", b), {})[s] = (R43[f"mlp_{b}_s{s}"], None)
    out = dict(graines=len(G), objets={}, comparaisons={}, glouton={}, elagage={}, qualite={}, durees={})

    def val(arch, nom, s, m):
        v = M.get((arch, nom), {}).get(s)
        if v is None:
            return None
        if m in v:
            return v[m]
        alias = {"E_cal": "V_cal", "E_fin": "V_fin", "E_max": "V_max", "V_cal": "E_cal", "V_fin": "E_fin", "V_max": "E_max"}
        return v.get(alias.get(m))

    # 1. par objet
    for (arch, nom), cs in sorted(OBJ.items()):
        gs = [s for s in G if s in cs]
        e = dict(n=len(gs))
        for m in ("E_cal", "E_fin", "E_max"):
            e[m] = moy(val(arch, nom, s, m) for s in gs)
        e["b_cal"] = moy((M[(arch, nom)][s].get("eb_cal", M[(arch, nom)][s].get("b_cal"))) for s in gs)
        rf, ao = zip(*[rappel_cal(*front_et_ancre(*cs[s])) for s in gs]) if gs else ((), ())
        e["Rf_cal"], e["Ao_cal"] = moy(rf), moy(ao)
        e["V_a"] = {str(c): moy(v_a(front_et_ancre(*cs[s])[0], c) for s in gs) for c in (0.85, 0.90, 0.92)}
        if nom in REGLES:
            for m in ("NB", "H0", "AMB", "crit"):
                e[m] = moy(M[(arch, nom)][s][m] for s in gs)
            e["NB_min_max"] = [int(min(M[(arch, nom)][s]["NB"] for s in gs)), int(max(M[(arch, nom)][s]["NB"] for s in gs))] if gs else None
            for c in ("desacc", "brier_moy", "brier_ens", "amb_chat"):
                e[c] = moy(cs[s][0]["fin_choix"]["regles"][nom]["diversite"][c] for s in gs)
        out["objets"][f"{arch}.{nom}"] = e
    # 2. comparaisons sans prédiction
    C = out["comparaisons"]
    cnn = [("cnn", r) for r in REGLES] + [("cnn", b) for b in ("LIVRE",) + RECETTES]
    for ref in (("cnn", "TOUS"), ("cnn", "SEUL10")):
        for x in cnn:
            if x == ref:
                continue
            for m in ("E_cal", "E_fin", "E_max"):
                C[f"{x[0]}.{x[1]}.{m}_moins_{ref[0]}.{ref[1]}"] = t_apparie(
                    [None if val(*x, s, m) is None or val(*ref, s, m) is None else val(*x, s, m) - val(*ref, s, m) for s in G])
            for c in (0.85, 0.90, 0.92):
                d = []
                for s in G:
                    a, b = v_a(front_et_ancre(*OBJ[x][s])[0], c), v_a(front_et_ancre(*OBJ[ref][s])[0], c)
                    d.append(None if a is None or b is None else a - b)
                C[f"{x[0]}.{x[1]}.V_a{c:.2f}_moins_{ref[0]}.{ref[1]}"] = t_apparie(d)
    # le convolutif contre le perceptron, bras par bras
    for a, b in ((("cnn", "TOUS"), ("mlp", "TOUS")), (("cnn", "GLOUTON"), ("mlp", "GLOUTON")),
                 (("cnn", "ELAGAGE"), ("mlp", "ELAGAGE")), (("cnn", "LIVRE"), ("mlp", "LIVRE")),
                 (("cnn", "GRAND"), ("mlp", "GRAND")), (("cnn", "SEUL"), ("mlp", "SEUL")),
                 (("cnn", "COURT"), ("mlp", "COURT2")), (("cnn", "SEUL10"), ("mlp", "TOUS")),
                 (("cnn", "SEUL10"), ("mlp", "COURT2")), (("cnn", "COURT"), ("mlp", "TOUS"))):
        for m in ("E_cal", "E_fin", "E_max"):
            C[f"{a[0]}.{a[1]}.{m}_moins_{b[0]}.{b[1]}"] = t_apparie(
                [None if val(*a, s, m) is None or val(*b, s, m) is None else val(*a, s, m) - val(*b, s, m) for s in G])
    # l'avance de l'ensemble sur chaque réseau seul, convolutif moins perceptron
    for bc, bm in (("SEUL10", "COURT2"), ("SEUL", "SEUL"), ("GRAND", "GRAND"), ("COURT", "COURT2"), ("LIVRE", "LIVRE")):
        d = []
        for s in G:
            xs = [val("cnn", "TOUS", s, "E_cal"), val("cnn", bc, s, "V_cal"), val("mlp", "TOUS", s, "E_cal"), val("mlp", bm, s, "V_cal")]
            d.append(None if None in xs else (xs[0] - xs[1]) - (xs[2] - xs[3]))
        C[f"avance_ensemble.cnn_{bc}_moins_mlp_{bm}"] = t_apparie(d)
    # ce que le réglage du seuil rapporte
    for x in cnn + [("mlp", "TOUS"), ("mlp", "COURT2"), ("mlp", "GRAND")]:
        C[f"reglage.{x[0]}.{x[1]}"] = t_apparie(
            [None if val(*x, s, "E_cal") is None else val(*x, s, "E_cal") - val(*x, s, "E_fin") for s in G])
    # la diversité de l'ensemble
    C["AMB.cnn.TOUS_moins_mlp.TOUS"] = t_apparie(
        [M[("cnn", "TOUS")][s]["AMB"] - M[("mlp", "TOUS")][s]["AMB"] for s in G if s in M.get(("mlp", "TOUS"), {})])
    C["NB.cnn.GLOUTON_moins_mlp.GLOUTON"] = t_apparie(
        [M[("cnn", "GLOUTON")][s]["NB"] - M[("mlp", "GLOUTON")][s]["NB"] for s in G if s in M.get(("mlp", "GLOUTON"), {})])
    # 3. la sélection gloutonne et l'élagage, sur le convolutif
    Cc = {s: D[f"cnn_INDEPC_s{s}"]["fin_choix"] for s in G}
    gl = out["glouton"]
    et = [Cc[s]["glouton"]["etape"] for s in G]
    gl["etape"] = dict(moyenne=float(np.mean(et)), min=int(min(et)), max=int(max(et)), a_64=int(sum(e == 64 for e in et)))
    gl["poids_max"] = float(np.mean([max(Cc[s]["regles"]["GLOUTON"]["m"]) / sum(Cc[s]["regles"]["GLOUTON"]["m"]) for s in G]))
    sel = {c: [] for c in ("lr", "wd", "eps", "V30")}
    aut = {c: [] for c in ("lr", "wd", "eps", "V30")}
    livre_retenu, corr = 0, []
    for s in G:
        m = np.array(Cc[s]["regles"]["GLOUTON"]["m"]) > 0
        rs = Cc[s]["reseaux"]
        for c in ("lr", "wd", "eps", "V30"):
            v = np.array(rs[c])
            f = np.log if c in ("lr", "wd") else (lambda x: x)
            sel[c].append(float(np.mean(f(v[m])))); aut[c].append(float(np.mean(f(v[~m]))) if (~m).any() else float("nan"))
        livre_retenu += int(m[D[f"cnn_INDEPC_s{s}"]["livre"]])
        corr.append(float(stats.pearsonr(Cc[s]["indiv"], rs["V30"])[0]))
    for c in ("lr", "wd", "eps", "V30"):
        a_, b_ = np.array(sel[c]), np.array(aut[c])
        ok = ~np.isnan(b_)
        conv = (lambda x: float(np.exp(x))) if c in ("lr", "wd") else (lambda x: float(x))
        gl[f"{c}_retenus"] = conv(a_.mean()); gl[f"{c}_autres"] = conv(b_[ok].mean()) if ok.any() else None
        gl[f"{c}_ecart"] = t_apparie(a_[ok] - b_[ok])
    gl["livre_retenu"] = livre_retenu
    gl["corr_critere_seul_verite_propre"] = dict(moyenne=float(np.mean(corr)), min=float(min(corr)), max=float(max(corr)))
    nbE = [32 - M[("cnn", "ELAGAGE")][s]["NB"] for s in G]
    out["elagage"]["retires"] = dict(moyenne=float(np.mean(nbE)), min=int(min(nbE)), max=int(max(nbE)))
    # 4. la qualité des 32 réseaux de la recherche aléatoire
    for arch, src in (("cnn", D), ("mlp", R45)):
        V = [np.array(src[f"{arch}_INDEPC_s{s}"]["fin_choix"]["reseaux"]["V30"]) for s in G if f"{arch}_INDEPC_s{s}" in src]
        out["qualite"][arch] = dict(moyenne=float(np.mean([v.mean() for v in V])), ecart_type=float(np.mean([v.std(ddof=1) for v in V])),
                                    min=float(np.mean([v.min() for v in V])), max=float(np.mean([v.max() for v in V])))
    # les durées
    for b in ("INDEPC",) + RECETTES:
        ds = [D[f"cnn_{b}_s{s}"]["duree"] / 3600 for s in GRAINES if f"cnn_{b}_s{s}" in D]
        out["durees"][b] = dict(n=len(ds), heures_moy=float(np.mean(ds)) if ds else None,
                               heures_tot=float(np.sum(ds)) if ds else None)
    json.dump(out, open(os.path.join(ICI, "ana46.json"), "w"), ensure_ascii=False, indent=1)
    for k, e in out["objets"].items():
        print(f"{k:<16} E_cal {e['E_cal']:.4f}  E_fin {e['E_fin']:.4f}  E_max {e['E_max']:.4f}  b {e['b_cal']:+.2f}  "
              f"Rf {e['Rf_cal']:.3f}  Ao {e['Ao_cal']:.3f}" + (f"  NB {e['NB']:.1f}  AMB {e['AMB']:.4f}  H0 {100 * e['H0']:+.2f}" if "NB" in e else ""))
    for k, v in C.items():
        if v:
            print(f"{k:<52} {100 * v['ecart']:+7.3f}  t = {v['t']:+6.2f}  IC [{100 * v['ic95'][0]:+.3f} ; "
                  f"{100 * v['ic95'][1]:+.3f}]  gagne {v['gagne']}/{v['n']}")
    print(json.dumps(out["glouton"], ensure_ascii=False)[:1200])
    print(out["elagage"], out["qualite"], out["durees"])
