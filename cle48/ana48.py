"""L'analyse descriptive de la clé 48, écrite avant les courses :
python3 ana48.py --cle47 DOSSIER [--resultats DOSSIER] [--sortie FICHIER]   (--cle47 est obligatoire)

Sans test : des moyennes, des écarts appariés par graine avec leur intervalle à 95 % (Student, 23 degrés de liberté),
aux trois seuils (réglé sur l'ancre, par défaut, meilleur seuil du juge — ce dernier, un diagnostic). Écrit ana48.json.
  1. les niveaux de chaque bras (ensemble, réseau seul, réseaux pris seuls) ;
  2. partie A : les écarts au réseau seul (SEUL10), aux trois seuils ;
  3. la Toile : chaque instantané face au réseau seul, les deux moyennes de poids (toutes, seconde moitié), le biais du
     chat réglé sur l'ancre de chacun ;
  4. la confiance (échelle des logits, plus grande probabilité, entropie, log-vraisemblance négative, probabilité du
     chat sur les chats) des réseaux seuls et des ensembles de la partie A ;
  5. partie B : la courbe du calcul — à tout le calcul (clé 47), au quart et au seizième : le réseau seul, K2, K4, leurs
     réseaux pris seuls, ce que la moyenne gagne sur eux, l'ambiguïté ; et les recettes égales d'une clé à l'autre (le
     réseau seul du quart et les réseaux de K4 de la clé 47 ; les réseaux de K2_Q et ceux de K8 ; ceux de K2_S et ceux
     de K32 ; ceux de K4_Q et le réseau seul du seizième) ;
  6. la décomposition des principaux écarts en rappel du chat et en précision des autres classes, chacun à son seuil
     réglé sur l'ancre ;
  7. les durées de calcul par bras.
"""
import json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import verdict48 as V


def ic(d):
    d = np.asarray(list(d), float)
    n = len(d)
    if n < 2:
        return dict(n=n, moy=float(d.mean()) if n else None)
    se = d.std(ddof=1) / math.sqrt(n)
    q = float(stats.t.ppf(0.975, n - 1))
    return dict(n=n, moy=float(d.mean()), ic95=[float(d.mean() - q * se), float(d.mean() + q * se)],
                gagne=int((d > 0).sum()), perd=int((d < 0).sum()))


def ecart(M, A, B):
    a, b = V.serie(M, A), V.serie(M, B)
    g = sorted(set(a) & set(b))
    return ic(a[s] - b[s] for s in g)


def niveau(M, X):
    return ic(V.serie(M, X).values())


def au_seuil(front, ancre):
    """Rappel du chat et précision des autres classes au seuil réglé sur l'ancre."""
    b = np.array(front["biais"]); note = np.array(ancre["note"])
    meilleurs = sorted(np.array(ancre["biais"])[note == note.max()])
    bc = float(meilleurs[(len(meilleurs) - 1) // 2])
    i = int(np.flatnonzero(b == bc)[0])
    return float(front["Rf"][i]), float(front["Ao"][i])


def decomposition(graines, quoi):
    """quoi(s) -> (front, ancre) d'une course ; rend {graine: (Rf, Ao)}."""
    out = {}
    for s in graines:
        try:
            f, a = quoi(s)
        except (KeyError, TypeError):
            continue
        out[s] = au_seuil(f, a)
    return out


ENS_A = ["K2_H", "TOILE4", "TOILE8"]
SEULS_A = ["SEUL_H", "PHOTO2", "MOY_TOILE4", "MOY_TOILE4_FIN", "MOY_TOILE8", "MOY_TOILE8_FIN", "TOILE4.DERNIER",
           "TOILE8.DERNIER", "MOY_K2_H"]
MEMBRES_A = ["K2_H.MEMBRES", "TOILE4.MEMBRES", "TOILE8.MEMBRES"]
ENS_B = ["K2_Q", "K4_Q", "K2_S", "K4_S", "K2_47", "K4_47"]
SEULS_B = ["SEUL_Q", "SEUL_S", "SEUL10_47", "MOY_K2_Q", "MOY_K4_Q", "MOY_K2_S", "MOY_K4_S"]
MEMBRES_B = ["K2_Q.MEMBRES", "K4_Q.MEMBRES", "K2_S.MEMBRES", "K4_S.MEMBRES", "K2_47.MEMBRES", "K4_47.MEMBRES",
             "K8_47.MEMBRES", "K32_47.MEMBRES"]


def main():
    D = V.lire_tranches(os.path.join(V.arg("--resultats", os.path.join(ICI, "resultats")), "tranche_J*.json"))
    assert "--cle47" in sys.argv, "--cle47 DOSSIER est obligatoire (les 24 fichiers réunis de la clé 47)"
    R47 = V.lire_tranches(None, V.EMPREINTES47, V.arg("--cle47", None))
    M = V.toutes_mesures(D, R47)
    out = {}
    # 1. les niveaux
    out["niveaux"] = {}
    for n in ENS_A + ENS_B:
        out["niveaux"][n] = {m: niveau(M, (n, m)) for m in ("E_cal", "E_fin", "E_max", "AMB", "AMB_chat")}
    for n in ["SEUL10"] + SEULS_A + SEULS_B + MEMBRES_A + MEMBRES_B:
        out["niveaux"][n] = {m: niveau(M, (n, m)) for m in ("V_cal", "V_fin", "V_max")}
    # 2. partie A : les écarts au réseau seul, aux trois seuils
    out["ecarts_au_seul"] = {}
    for n, m in [(x, "E") for x in ENS_A] + [(x, "V") for x in SEULS_A + MEMBRES_A]:
        out["ecarts_au_seul"][n] = {s: ecart(M, (n, f"{m}_{s}"), ("SEUL10", f"V_{s}")) for s in ("cal", "fin", "max")}
    # 3. la Toile : chaque instantané face au réseau seul ; les deux moyennes ; les biais réglés
    toile = {}
    for c in (4, 8):
        inst = {}
        for i in range(c):
            a = {s: v["V_cal_tous"][i] for s, v in M.get(f"TOILE{c}.MEMBRES", {}).items()}
            b = V.serie(M, ("SEUL10", "V_cal"))
            g = sorted(set(a) & set(b))
            inst[str(i + 1)] = ic(a[s] - b[s] for s in g)
        toile[f"TOILE{c}"] = dict(
            instantanes_moins_seul=inst,
            moyenne_fin_moins_moyenne=ecart(M, (f"MOY_TOILE{c}_FIN", "V_cal"), (f"MOY_TOILE{c}", "V_cal")),
            moyenne_moins_ensemble=ecart(M, (f"MOY_TOILE{c}", "V_cal"), (f"TOILE{c}", "E_cal")),
            moyenne_moins_dernier=ecart(M, (f"MOY_TOILE{c}", "V_cal"), (f"TOILE{c}.DERNIER", "V_cal")))
    toile["MOY_TOILE8_moins_MOY_TOILE4"] = ecart(M, ("MOY_TOILE8", "V_cal"), ("MOY_TOILE4", "V_cal"))
    toile["TOILE8_moins_TOILE4"] = ecart(M, ("TOILE8", "E_cal"), ("TOILE4", "E_cal"))
    toile["biais_regle"] = {n: niveau(M, (n, "b_cal")) for n in ("SEUL10", "SEUL_H", "PHOTO2", "MOY_TOILE4",
                                                                 "MOY_TOILE4_FIN", "MOY_TOILE8", "MOY_TOILE8_FIN",
                                                                 "TOILE4.DERNIER", "TOILE8.DERNIER")}
    toile["biais_regle"].update({n: niveau(M, (n, "eb_cal")) for n in ("TOILE4", "TOILE8", "K2_H")})
    out["toile"] = toile
    # 4. la confiance (partie A)
    conf = {}
    for n in ("SEUL10", "SEUL_H", "PHOTO2", "MOY_TOILE4", "MOY_TOILE4_FIN", "MOY_TOILE8", "MOY_TOILE8_FIN",
              "TOILE4.DERNIER", "TOILE8.DERNIER"):
        conf[n] = {m: niveau(M, (n, m)) for m in ("LOG", "CONF", "ENT", "NLL", "PCHAT")}
    ens_conf = {}
    for s in V.GRAINES_A:
        for c in (4, 8):
            r = D.get(f"cnn_TOILE{c}_s{s}")
            if r:
                ens_conf.setdefault(f"TOILE{c}", []).append(r["fin_confiance"]["ens"])
        r = D.get(f"cnn_PHOTO2_s{s}")
        if r:
            ens_conf.setdefault("K2_H", []).append(r["fin_confiance"]["ens"])
    for n, L in ens_conf.items():
        conf[n] = {m: ic(x[m] for x in L) for m in ("ecart_logits", "conf", "entropie", "nll", "p_chat_chats")}
    conf["LOG_SEUL10_moins"] = {n: ecart(M, ("SEUL10", "LOG"), (n, "LOG")) for n in
                                ("SEUL_H", "PHOTO2", "MOY_TOILE4", "MOY_TOILE4_FIN", "MOY_TOILE8", "MOY_TOILE8_FIN",
                                 "TOILE4.DERNIER", "TOILE8.DERNIER")}
    out["confiance"] = conf
    # 5. partie B : la courbe du calcul
    courbe = {}
    for part, seul, k2, k4 in (("1", "SEUL10_47", "K2_47", "K4_47"), ("1/4", "SEUL_Q", "K2_Q", "K4_Q"),
                               ("1/16", "SEUL_S", "K2_S", "K4_S")):
        courbe[part] = dict(seul=niveau(M, (seul, "V_cal")))
        for k in (k2, k4):
            courbe[part][k] = dict(ensemble=niveau(M, (k, "E_cal")), membres=niveau(M, (f"{k}.MEMBRES", "V_cal")),
                                   gain_sur_membres=ecart(M, (k, "E_cal"), (f"{k}.MEMBRES", "V_cal")),
                                   ecart_au_seul=ecart(M, (k, "E_cal"), (seul, "V_cal")),
                                   membres_moins_seul=ecart(M, (f"{k}.MEMBRES", "V_cal"), (seul, "V_cal")),
                                   amb=niveau(M, (k, "AMB")))
    out["courbe_calcul"] = courbe
    out["recettes_egales"] = {
        "SEUL_Q-K4_47.MEMBRES": ecart(M, ("SEUL_Q", "V_cal"), ("K4_47.MEMBRES", "V_cal")),
        "K2_Q.MEMBRES-K8_47.MEMBRES": ecart(M, ("K2_Q.MEMBRES", "V_cal"), ("K8_47.MEMBRES", "V_cal")),
        "K2_S.MEMBRES-K32_47.MEMBRES": ecart(M, ("K2_S.MEMBRES", "V_cal"), ("K32_47.MEMBRES", "V_cal")),
        "K4_Q.MEMBRES-SEUL_S": ecart(M, ("K4_Q.MEMBRES", "V_cal"), ("SEUL_S", "V_cal")),
    }
    out["derivees"] = {n: niveau(M, (n, "d")) for n in ("GAIN.K2_47", "GAIN.K2_Q", "GAIN.K2_S", "GMEM.K2_47",
                                                        "GMEM.K2_Q", "GMEM.K2_S", "PRIX.Q", "PRIX.S")}
    # 6. la décomposition des écarts principaux (rappel du chat, autres classes), chacun à son seuil réglé
    def f_ens(b, cle="fin_ens"):
        return lambda s: (D[f"cnn_{b}_s{s}"][cle]["front"], D[f"cnn_{b}_s{s}"][cle]["ancre"])

    def f_un(b, cle):
        return lambda s: (D[f"cnn_{b}_s{s}"][cle], D[f"cnn_{b}_s{s}"][cle]["ancre"])
    paires = {
        "MOY_TOILE4-SEUL10": (V.GRAINES_A, f_un("TOILE4", "fin_moyenne"), f_un("SEUL10", "fin_livre")),
        "MOY_TOILE8-SEUL10": (V.GRAINES_A, f_un("TOILE8", "fin_moyenne"), f_un("SEUL10", "fin_livre")),
        "TOILE4-SEUL10": (V.GRAINES_A, f_ens("TOILE4"), f_un("SEUL10", "fin_livre")),
        "PHOTO2-SEUL10": (V.GRAINES_A, lambda s: (D[f"cnn_PHOTO2_s{s}"]["photo"]["fin_livre"],
                                                  D[f"cnn_PHOTO2_s{s}"]["photo"]["fin_livre"]["ancre"]),
                          f_un("SEUL10", "fin_livre")),
        "K2_H-SEUL10": (V.GRAINES_A, f_ens("PHOTO2"), f_un("SEUL10", "fin_livre")),
        "K2_Q-SEUL_Q": (V.GRAINES_B, f_ens("K2_Q"), f_un("SEUL_Q", "fin_livre")),
        "K2_S-SEUL_S": (V.GRAINES_B, f_ens("K2_S"), f_un("SEUL_S", "fin_livre")),
        "K4_S-SEUL_S": (V.GRAINES_B, f_ens("K4_S"), f_un("SEUL_S", "fin_livre")),
    }
    out["decomposition"] = {}
    for n, (graines, fa, fb) in paires.items():
        da, db = decomposition(graines, fa), decomposition(graines, fb)
        g = sorted(set(da) & set(db))
        if g:
            out["decomposition"][n] = dict(
                rappel_chat=ic(da[s][0] - db[s][0] for s in g),
                precision_autres=ic(da[s][1] - db[s][1] for s in g),
                part_chat_dans_V=ic(V.W_CHAT * (da[s][0] - db[s][0]) for s in g),
                part_autres_dans_V=ic((1 - V.W_CHAT) * (da[s][1] - db[s][1]) for s in g))
    # 7. les durées
    dur = {}
    for k, r in D.items():
        dur.setdefault(r["tache"]["bras"], []).append(r.get("duree", float("nan")) / 3600)
    out["durees_heures"] = {b: dict(n=len(v), moy=float(np.mean(v)), mediane=float(np.median(v)),
                                    max=float(np.max(v)), total=float(np.sum(v))) for b, v in dur.items()}
    out["meme_monde"] = V.meme_monde(D)
    json.dump(out, open(V.arg("--sortie", os.path.join(ICI, "ana48.json")), "w"), ensure_ascii=False, indent=1)
    # le résumé
    print("Partie A — écarts au réseau seul (SEUL10), à seuil égal (réglé sur l'ancre), en points :")
    for n, e in out["ecarts_au_seul"].items():
        c = e["cal"]
        if c.get("n", 0) >= 2:
            print(f"  {n:<16} {100 * c['moy']:+7.2f}  [{100 * c['ic95'][0]:+6.2f} ; {100 * c['ic95'][1]:+6.2f}]  "
                  f"({c['gagne']}/{c['n']})")
    print("Partie B — la courbe du calcul (réseau seul ; K2, K4 : ensemble, réseaux seuls, gain de la moyenne, "
          "ambiguïté) :")
    for part, c in courbe.items():
        s = c["seul"]
        if s.get("n", 0) < 1:
            continue
        ligne = f"  calcul {part:>4} : seul {s['moy']:.4f}"
        for k, x in c.items():
            if k == "seul" or x["ensemble"].get("n", 0) < 1:
                continue
            ligne += (f" | {k} {x['ensemble']['moy']:.4f} réseaux {x['membres']['moy']:.4f} gain "
                      f"{100 * x['gain_sur_membres']['moy']:+.2f} amb {x['amb']['moy']:.4f}")
        print(ligne)
    print("recettes égales :", {k: (round(100 * v["moy"], 2) if v.get("moy") is not None else None)
                                for k, v in out["recettes_egales"].items()})
    print("durées (heures, médianes) :", {b: round(v["mediane"], 2) for b, v in out["durees_heures"].items()})
    print("même monde :", out["meme_monde"])


if __name__ == "__main__":
    main()
