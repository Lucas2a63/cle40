"""L'analyse descriptive de la clé 47, écrite avant les courses :
python3 ana47.py [--cle46 DOSSIER] [--resultats DOSSIER] [--sortie FICHIER]

Sans test : des moyennes, des écarts appariés par graine avec leur intervalle à 95 % (Student, 23 degrés de liberté),
aux trois seuils (réglé sur l'ancre, par défaut, meilleur seuil du juge — ce dernier, un diagnostic). Écrit ana47.json.
  1. les niveaux de chaque bras (ensemble, réseau seul, réseaux pris seuls) ;
  2. les écarts au réseau seul (SEUL10, le réseau du VEILLEUR) ;
  3. la courbe du découpage : k = 1 (SEUL10), 2, 4, 8, 32 — l'ensemble, ses réseaux pris seuls (moyenne), ce que la
     moyenne gagne sur eux, l'ambiguïté ;
  4. les règles du choix final de POPC face à celles de la recherche aléatoire de la clé 46 (INDEPC) ;
  5. les soupes : celle des enfants (uniforme et gourmande), les moyennes de poids sans ancêtre commun (K2 à K32, et les
     instantanés de la TOILE) ;
  6. la décomposition des principaux écarts en rappel du chat et en précision des autres classes, chacun à son seuil
     réglé sur l'ancre ;
  7. les durées de calcul par bras.
"""
import json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import verdict47 as V


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


def decomposition(D, quoi):
    """quoi(r) -> (front, ancre) d'une course ; rend {graine: (Rf, Ao)}."""
    out = {}
    for s in V.GRAINES:
        try:
            f, a = quoi(s)
        except (KeyError, TypeError):
            continue
        out[s] = au_seuil(f, a)
    return out


def main():
    D = V.lire_tranches(os.path.join(V.arg("--resultats", os.path.join(ICI, "resultats")), "tranche_J*.json"))
    R46 = V.lire_tranches(None, V.EMPREINTES46, V.arg("--cle46", os.path.join(ICI, "..", "cle46", "resultats")))
    M = V.toutes_mesures(D, R46)
    out = {}
    # 1. les niveaux
    ens = ["K2", "K4", "K8", "K32", "TOILE", "SOUPE.ENFANTS"] + [f"POPC.{r}" for r in V.REGLES] + \
          [f"INDEPC.{r}" for r in V.REGLES]
    seuls = ["SEUL10", "SWA", "EMA", "PHOTO", "SOUPE", "GOURMANDE", "TOILE.DERNIER", "POPC.LIVRE", "SEUL10_46",
             "SEUL_46", "COURT_46", "GRAND_46", "MOY_K2", "MOY_K4", "MOY_K8", "MOY_K32", "MOY_TOILE"]
    membres = ["K2.MEMBRES", "K4.MEMBRES", "K8.MEMBRES", "K32.MEMBRES", "TOILE.MEMBRES", "SOUPE.MEMBRES"]
    out["niveaux"] = {}
    for n in ens:
        out["niveaux"][n] = {m: niveau(M, (n, m)) for m in ("E_cal", "E_fin", "E_max", "AMB", "AMB_chat")}
    for n in seuls + membres:
        out["niveaux"][n] = {m: niveau(M, (n, m)) for m in ("V_cal", "V_fin", "V_max")}
    # 2. les écarts au réseau seul, aux trois seuils
    out["ecarts_au_seul"] = {}
    for n, m in [(x, "E") for x in ens if not x.startswith("INDEPC")] + [(x, "V") for x in seuls if x != "SEUL10"] + \
                [(x, "V") for x in membres]:
        out["ecarts_au_seul"][n] = {s: ecart(M, (n, f"{m}_{s}"), ("SEUL10", f"V_{s}")) for s in ("cal", "fin", "max")}
    # 3. la courbe du découpage
    courbe = {"1": dict(ensemble=niveau(M, ("SEUL10", "V_cal")))}
    for k in (2, 4, 8, 32):
        n = f"K{k}"
        courbe[str(k)] = dict(ensemble=niveau(M, (n, "E_cal")), membres=niveau(M, (f"{n}.MEMBRES", "V_cal")),
                              meilleur_membre=niveau(M, (f"{n}.MEMBRES", "V_cal_max")),
                              gain_sur_membres=ecart(M, (n, "E_cal"), (f"{n}.MEMBRES", "V_cal")),
                              ecart_au_seul=ecart(M, (n, "E_cal"), ("SEUL10", "V_cal")),
                              membres_moins_seul=ecart(M, (f"{n}.MEMBRES", "V_cal"), ("SEUL10", "V_cal")),
                              amb=niveau(M, (n, "AMB")))
    out["courbe_decoupage"] = courbe
    # 4. POPC face à INDEPC (clé 46), règle par règle
    out["popc_contre_indepc"] = {r: ecart(M, (f"POPC.{r}", "E_cal"), (f"INDEPC.{r}", "E_cal")) for r in V.REGLES}
    out["popc_regles_contre_tous"] = {r: ecart(M, (f"POPC.{r}", "E_cal"), ("POPC.TOUS", "E_cal")) for r in V.REGLES}
    regl = {"lr": [], "wd": [], "V30": []}
    for s in V.GRAINES:
        r = D.get(f"cnn_POPC_s{s}")
        if r:
            c = r["fin_choix"]["reseaux"]
            regl["lr"] += c["lr"]; regl["wd"] += c["wd"]; regl["V30"] += c["V30"]
    if regl["lr"]:
        lr, v = np.log(regl["lr"]), np.array(regl["V30"])
        out["popc_taux"] = dict(pente_V30_par_log_lr=float(np.polyfit(lr, v, 1)[0]), n=len(v),
                                V30_moyen=float(v.mean()), V30_ecart_type=float(v.std()))
    # 5. les soupes
    garde = [sum(D[f"cnn_SOUPE_s{s}"]["fin_gourmande"]["m"]) for s in V.GRAINES if f"cnn_SOUPE_s{s}" in D]
    out["soupes"] = dict(soupe_moins_enfants=ecart(M, ("SOUPE", "V_cal"), ("SOUPE.MEMBRES", "V_cal")),
                         gourmande_moins_soupe=ecart(M, ("GOURMANDE", "V_cal"), ("SOUPE", "V_cal")),
                         enfants_gardes_par_la_gourmande=ic(garde) if garde else None,
                         sans_ancetre={n: niveau(M, (n, "V_cal")) for n in ("MOY_K2", "MOY_K4", "MOY_K8", "MOY_K32",
                                                                          "MOY_TOILE")})
    # 6. la décomposition des écarts principaux (rappel du chat, autres classes), chacun à son seuil réglé
    def f_ens(b, cle="fin_ens"):
        return lambda s: (D[f"cnn_{b}_s{s}"][cle]["front"], D[f"cnn_{b}_s{s}"][cle]["ancre"])

    def f_un(b, cle):
        return lambda s: (D[f"cnn_{b}_s{s}"][cle], D[f"cnn_{b}_s{s}"][cle]["ancre"])
    seul = decomposition(D, f_un("VEILLEUR", "fin_livre"))
    autres = dict(K4=decomposition(D, f_ens("K4P")), K32=decomposition(D, f_ens("K32")),
                  K2=decomposition(D, f_ens("K2")), K8=decomposition(D, f_ens("K8")),
                  SOUPE=decomposition(D, f_un("SOUPE", "fin_soupe")), TOILE=decomposition(D, f_ens("TOILE")),
                  PHOTO=decomposition(D, lambda s: (D[f"cnn_K4P_s{s}"]["photo"]["fin_livre"],
                                                    D[f"cnn_K4P_s{s}"]["photo"]["fin_livre"]["ancre"])))
    out["decomposition_contre_seul"] = {}
    for n, d in autres.items():
        g = sorted(set(d) & set(seul))
        if g:
            out["decomposition_contre_seul"][n] = dict(
                rappel_chat=ic(d[s][0] - seul[s][0] for s in g),
                precision_autres=ic(d[s][1] - seul[s][1] for s in g),
                part_chat_dans_V=ic(V.W_CHAT * (d[s][0] - seul[s][0]) for s in g),
                part_autres_dans_V=ic((1 - V.W_CHAT) * (d[s][1] - seul[s][1]) for s in g))
    # 7. les durées
    dur = {}
    for k, r in D.items():
        dur.setdefault(r["tache"]["bras"], []).append(r.get("duree", float("nan")) / 3600)
    out["durees_heures"] = {b: dict(n=len(v), moy=float(np.mean(v)), max=float(np.max(v)), total=float(np.sum(v)))
                            for b, v in dur.items()}
    out["identites"] = V.identites(D, R46)
    json.dump(out, open(V.arg("--sortie", os.path.join(ICI, "ana47.json")), "w"), ensure_ascii=False, indent=1)
    # le résumé
    print("Écarts au réseau seul (SEUL10), à seuil égal (réglé sur l'ancre), en points :")
    for n, e in out["ecarts_au_seul"].items():
        c = e["cal"]
        if c.get("n", 0) >= 2:
            print(f"  {n:<16} {100 * c['moy']:+7.2f}  [{100 * c['ic95'][0]:+6.2f} ; {100 * c['ic95'][1]:+6.2f}]  "
                  f"({c['gagne']}/{c['n']})")
    print("Courbe du découpage (k : ensemble, réseaux seuls, gain de la moyenne, ambiguïté) :")
    for k, c in courbe.items():
        e = c["ensemble"]
        if e.get("n", 0) < 1:
            continue
        if k == "1":
            print(f"  k = {k:>2} : {e['moy']:.4f}")
        else:
            print(f"  k = {k:>2} : {e['moy']:.4f}  réseaux {c['membres']['moy']:.4f}  gain "
                  f"{100 * c['gain_sur_membres']['moy']:+.2f}  amb {c['amb']['moy']:.4f}")
    print("durées (heures) :", {b: round(v["moy"], 2) for b, v in out["durees_heures"].items()})
    print("identités :", out["identites"])


if __name__ == "__main__":
    main()
