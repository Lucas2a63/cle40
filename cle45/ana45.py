"""Analyse de la clé 45 (descriptive, et comparaisons sans prédiction, signalées comme telles) :
python3 ana45.py [--references DOSSIER] → ana45.json

1. Par règle de choix : les mesures de verdict45 (E_cal, E_fin, E_max, H0, nombre de réseaux distincts, critère sur
   les ancres réunies) et la diversité de l'ensemble choisi sur le juge (ambiguïté, désaccord, Brier moyen et de
   l'ensemble) ; le rappel du chat et la précision ordinaire au seuil réglé ; l'ensemble à rappel du chat égal (0,85,
   0,90, 0,92).
2. Comparaisons sans prédiction : chaque règle contre TOUS, et contre le grand réseau, le réseau court réglé et le
   réseau seul de la clé 43, à seuil égal, au seuil par défaut et au meilleur seuil ; les règles entre elles à rappel
   du chat égal.
3. Ce que choisit la sélection gloutonne : l'étape retenue ; le poids du réseau le plus lourd ; les réglages (taux,
   décroissance, force de curation) et la vérité propre au seuil par défaut des réseaux retenus et des autres ; si le
   réseau livré (le mieux noté par la dernière ancre) est retenu ; la corrélation, sur les 32 réseaux d'une course,
   entre le critère pris seul (sur l'ancre) et la vérité propre (sur le juge).
4. L'élagage : combien de réseaux il retire.
Écrit avant toute course du plan : n'a rien affiché des données de la clé 45.
"""
import glob, json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
from verdict45 import GRAINES, REGLES, W_CHAT, lire_front, mesures_regle, mesures_grand, charger


def t_apparie(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("nan")
    q = stats.t.ppf(0.975, len(d) - 1)
    return dict(n=len(d), ecart=float(d.mean()), t=float(t), ic95=[float(d.mean() - q * se), float(d.mean() + q * se)],
                gagne=int((d > 0).sum()), perd=int((d < 0).sum()))


def ao_a(front, cible):
    Rf, Ao = np.array(front["Rf"]), np.array(front["Ao"])
    o = np.argsort(Rf, kind="stable")
    return float(np.interp(cible, Rf[o], Ao[o])) if Rf.min() <= cible <= Rf.max() else None


if __name__ == "__main__":
    dref = sys.argv[sys.argv.index("--references") + 1] if "--references" in sys.argv else \
        os.path.join(ICI, "..", "cle43", "resultats")
    D, R = charger(dref)
    G = [s for s in GRAINES if f"mlp_INDEPC_s{s}" in D]
    C = {s: D[f"mlp_INDEPC_s{s}"]["fin_choix"] for s in G}
    M = {s: {nom: mesures_regle(C[s]["regles"][nom]) for nom in REGLES} for s in G}
    REF = {b: {s: mesures_grand(R[f"mlp_{b}_s{s}"]) for s in G} for b in ("GRAND", "COURT2", "SEUL")}
    out = dict(graines=len(G), regles={}, comparaisons={}, glouton={}, elagage={})
    # 1. par règle
    for nom in REGLES:
        e = {c: float(np.mean([M[s][nom][c] for s in G])) for c in ("E_cal", "E_fin", "E_max", "H0", "NB", "crit", "AMB")}
        e.update({c + "_sd": float(np.std([M[s][nom][c] for s in G], ddof=1)) for c in ("E_cal", "NB")})
        e["NB_min_max"] = [int(min(M[s][nom]["NB"] for s in G)), int(max(M[s][nom]["NB"] for s in G))]
        for c in ("desacc", "brier_moy", "brier_ens", "amb_chat"):
            e[c] = float(np.mean([C[s]["regles"][nom]["diversite"][c] for s in G]))
        rf, ao = [], []
        for s in G:
            g = C[s]["regles"][nom]
            i = g["front"]["biais"].index(M[s][nom]["eb_cal"])
            rf.append(g["front"]["Rf"][i]); ao.append(g["front"]["Ao"][i])
        e["Rf_cal"], e["Ao_cal"] = float(np.mean(rf)), float(np.mean(ao))
        e["Ao_a"] = {c: float(np.mean([ao_a(C[s]["regles"][nom]["front"], c) for s in G])) for c in (0.85, 0.90, 0.92)}
        out["regles"][nom] = e
    # 2. comparaisons sans prédiction
    for nom in REGLES:
        if nom != "TOUS":
            for c in ("E_cal", "E_fin", "E_max", "H0", "AMB"):
                out["comparaisons"][f"{nom}.{c}_moins_TOUS"] = t_apparie([M[s][nom][c] - M[s]["TOUS"][c] for s in G])
            for c in (0.85, 0.90, 0.92):
                d = [ao_a(C[s]["regles"][nom]["front"], c) - ao_a(C[s]["regles"]["TOUS"]["front"], c) for s in G]
                out["comparaisons"][f"{nom}.Ao_{c:.2f}_moins_TOUS"] = t_apparie(d)
        for b in ("GRAND", "COURT2", "SEUL"):
            for (me, mr) in (("E_cal", "V_cal"), ("E_fin", "V_fin"), ("E_max", "V_max")):
                out["comparaisons"][f"{nom}.{me}_moins_{b}.{mr}"] = t_apparie([M[s][nom][me] - REF[b][s][mr] for s in G])
    for a, b in (("GLOUTON", "ELAGAGE"), ("GLOUTON", "MEILLEURS8"), ("TETE", "HASARD"), ("ELAGAGE", "MEILLEURS16"),
                 ("TETE", "TOUS"), ("MEILLEURS16", "HASARD")):
        out["comparaisons"][f"{a}.E_cal_moins_{b}"] = t_apparie([M[s][a]["E_cal"] - M[s][b]["E_cal"] for s in G])
    # 3. ce que choisit la sélection gloutonne
    gl = out["glouton"]
    gl["etape"] = dict(moyenne=float(np.mean([C[s]["glouton"]["etape"] for s in G])),
                       min=int(min(C[s]["glouton"]["etape"] for s in G)), max=int(max(C[s]["glouton"]["etape"] for s in G)))
    gl["poids_max"] = float(np.mean([max(C[s]["regles"]["GLOUTON"]["m"]) / sum(C[s]["regles"]["GLOUTON"]["m"]) for s in G]))
    sel = {c: [] for c in ("lr", "wd", "eps", "V30")}
    aut = {c: [] for c in ("lr", "wd", "eps", "V30")}
    livre_retenu, corr = 0, []
    for s in G:
        m = np.array(C[s]["regles"]["GLOUTON"]["m"]) > 0
        rs = C[s]["reseaux"]
        for c in ("lr", "wd", "eps", "V30"):
            v = np.array(rs[c])
            f = np.log if c in ("lr", "wd") else (lambda x: x)
            sel[c].append(float(np.mean(f(v[m])))); aut[c].append(float(np.mean(f(v[~m]))) if (~m).any() else float("nan"))
        livre_retenu += int(m[D[f"mlp_INDEPC_s{s}"]["livre"]])
        corr.append(float(stats.pearsonr(C[s]["indiv"], rs["V30"])[0]))
    for c in ("lr", "wd", "eps", "V30"):
        a_, b_ = np.array(sel[c]), np.array(aut[c])
        ok = ~np.isnan(b_)
        conv = (lambda x: float(np.exp(x))) if c in ("lr", "wd") else (lambda x: float(x))
        gl[f"{c}_retenus"] = conv(a_.mean()); gl[f"{c}_autres"] = conv(b_[ok].mean())
        gl[f"{c}_ecart"] = t_apparie(a_[ok] - b_[ok])
    gl["livre_retenu"] = livre_retenu
    gl["corr_critere_seul_verite_propre"] = dict(moyenne=float(np.mean(corr)), min=float(min(corr)), max=float(max(corr)))
    # 4. l'élagage
    out["elagage"]["retires"] = dict(moyenne=float(np.mean([32 - M[s]["ELAGAGE"]["NB"] for s in G])),
                                    min=int(min(32 - M[s]["ELAGAGE"]["NB"] for s in G)),
                                    max=int(max(32 - M[s]["ELAGAGE"]["NB"] for s in G)))
    json.dump(out, open(os.path.join(ICI, "ana45.json"), "w"), ensure_ascii=False, indent=1)
    for nom, e in out["regles"].items():
        print(f"{nom:<12} E_cal {e['E_cal']:.4f}  E_fin {e['E_fin']:.4f}  E_max {e['E_max']:.4f}  H0 {100 * e['H0']:+.2f}  "
              f"NB {e['NB']:5.1f} [{e['NB_min_max'][0]}–{e['NB_min_max'][1]}]  AMB {e['AMB']:.4f}  Rf {e['Rf_cal']:.3f}  "
              f"Ao(0,90) {e['Ao_a'][0.90]:.4f}")
    for k, v in out["comparaisons"].items():
        print(f"{k:<40} {100 * v['ecart']:+7.3f}  t = {v['t']:+6.2f}  IC [{100 * v['ic95'][0]:+.3f} ; "
              f"{100 * v['ic95'][1]:+.3f}]  gagne {v['gagne']}/{v['n']}")
    print(json.dumps(out["glouton"], ensure_ascii=False)[:1500])
    print(out["elagage"])
