"""Clé 46 — analyses sans prédiction, écrites PENDANT la campagne (le 22 octobre vers 3 h 30, heure de Paris), avant
qu'aucune course du plan ne soit finie et avant le verdict ; déposées dans le dépôt avant le verdict pour les dater.
Tout ce qu'elles produisent sera rapporté comme « après coup » (non pré-enregistré, sans correction).
python3 apres46.py [--cle43 DOSSIER] [--cle45 DOSSIER] → apres46.json

1. D'où vient chaque écart à seuil égal (un diagnostic, pas une cause) : au seuil réglé sur l'ancre de chacun, l'écart
   de vérité ΔV = w·ΔRf + (1 − w)·ΔAo se partage entre le rappel du chat (w·ΔRf) et la précision sur les autres
   classes ((1 − w)·ΔAo), apparié par graine — pour chaque comparaison pré-enregistrée de vérité (P1–P4, S1–S10).
2. Ce que la moyenne gagne sur ses propres réseaux, en vérité, au seuil par défaut (le seul où la vérité propre des 32
   réseaux est mesurée) : l'ensemble des 32 moins la moyenne de ses réseaux, moins le meilleur d'entre eux selon le
   juge (un oracle), moins le réseau livré (le mieux noté par l'ancre) ; sur le convolutif et sur le perceptron (clé
   45), et la différence convolutif − perceptron, appariée par graine.
3. La décomposition du score de Brier de l'ensemble des 32 sur le juge (une identité comptable : Brier de l'ensemble =
   Brier moyen de ses réseaux − ambiguïté) : convolutif − perceptron, terme par terme, apparié ; et le désaccord.
"""
import json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
from verdict46 import GRAINES, W_CHAT, charger, toutes_mesures, arg


def t_apparie(d):
    d = np.asarray([x for x in d if x is not None], float)
    if len(d) < 3:
        return None
    se = d.std(ddof=1) / math.sqrt(len(d))
    q = stats.t.ppf(0.975, len(d) - 1)
    return dict(n=len(d), ecart=float(d.mean()), t=float(d.mean() / se) if se > 0 else float("nan"),
                ic95=[float(d.mean() - q * se), float(d.mean() + q * se)], gagne=int((d > 0).sum()))


def au_seuil(front, ancre):
    """(Rf, Ao) du juge au biais réglé sur l'ancre (médiane basse des biais qui maximisent la note de l'ancre)."""
    note = np.array(ancre["note"])
    meilleurs = sorted(np.array(ancre["biais"])[note == note.max()])
    bc = float(meilleurs[(len(meilleurs) - 1) // 2])
    i = front["biais"].index(bc)
    return float(front["Rf"][i]), float(front["Ao"][i])


if __name__ == "__main__":
    D, R45, R43 = charger(arg("--cle43", os.path.join(ICI, "..", "cle43", "resultats")),
                          arg("--cle45", os.path.join(ICI, "..", "cle45", "resultats")))
    M = toutes_mesures(D, R45, R43)

    def objet(arch, nom, s):
        """(front, ancre) de l'objet (arch, nom) à la graine s, ou None."""
        src = D if arch == "cnn" else (R45 if nom in ("TOUS", "GLOUTON", "LIVRE") else R43)
        if nom in ("TOUS", "GLOUTON"):
            c = src.get(f"{arch}_INDEPC_s{s}")
            if c is None:
                return None
            g = c["fin_choix"]["regles"][nom]
            return g["front"], g["ancre"]
        c = src.get(f"{arch}_INDEPC_s{s}" if nom == "LIVRE" else f"{arch}_{nom}_s{s}")
        if c is None:
            return None
        return c["fin_livre"], c["fin_livre"]["ancre"]

    out = dict(decomposition={}, gain_sur_ses_reseaux={}, brier={})
    # 1. la décomposition des écarts à seuil égal
    PAIRES = {"P1": (("cnn", "TOUS"), ("cnn", "SEUL10")), "P2": (("cnn", "TOUS"), ("cnn", "GRAND")),
              "P3": (("cnn", "SEUL10"), ("mlp", "TOUS")), "S1": (("cnn", "TOUS"), ("cnn", "SEUL")),
              "S2": (("cnn", "TOUS"), ("cnn", "COURT")), "S3": (("cnn", "GLOUTON"), ("cnn", "TOUS")),
              "S6": (("cnn", "TOUS"), ("mlp", "TOUS")), "S7": (("cnn", "SEUL10"), ("cnn", "SEUL")),
              "S8": (("cnn", "GLOUTON"), ("cnn", "SEUL10")), "S9": (("cnn", "TOUS"), ("cnn", "LIVRE")),
              "S10": (("cnn", "SEUL10"), ("cnn", "GRAND")),
              "mlp_TOUS_moins_COURT2": (("mlp", "TOUS"), ("mlp", "COURT2"))}
    for nom, (a, b) in PAIRES.items():
        dR, dA, dV = [], [], []
        for s in GRAINES:
            oa, ob = objet(*a, s), objet(*b, s)
            if oa is None or ob is None:
                continue
            (ra, aa), (rb, ab) = au_seuil(*oa), au_seuil(*ob)
            dR.append(W_CHAT * (ra - rb)); dA.append((1 - W_CHAT) * (aa - ab))
            dV.append(W_CHAT * (ra - rb) + (1 - W_CHAT) * (aa - ab))
        out["decomposition"][nom] = dict(ecart=t_apparie(dV), part_chat=t_apparie(dR), part_autres=t_apparie(dA))
    # P4 : la différence des deux avances, partagée de même
    dR, dA = [], []
    for s in GRAINES:
        o = [objet("mlp", "TOUS", s), objet("mlp", "COURT2", s), objet("cnn", "TOUS", s), objet("cnn", "SEUL10", s)]
        if None in o:
            continue
        (r1, a1), (r2, a2), (r3, a3), (r4, a4) = [au_seuil(*x) for x in o]
        dR.append(W_CHAT * ((r1 - r2) - (r3 - r4))); dA.append((1 - W_CHAT) * ((a1 - a2) - (a3 - a4)))
    out["decomposition"]["P4"] = dict(ecart=t_apparie([x + y for x, y in zip(dR, dA)]), part_chat=t_apparie(dR),
                                      part_autres=t_apparie(dA))
    # 2. ce que la moyenne gagne sur ses propres réseaux, au seuil par défaut
    par = {}
    for arch, src in (("cnn", D), ("mlp", R45)):
        for s in GRAINES:
            c = src.get(f"{arch}_INDEPC_s{s}")
            if c is None:
                continue
            V = np.array(c["fin_choix"]["reseaux"]["V30"])
            E0 = M[(arch, "TOUS")][s]["E_fin"]
            par.setdefault(arch, {})[s] = dict(moins_moyenne=E0 - float(V.mean()), moins_meilleur=E0 - float(V.max()),
                                               moins_livre=E0 - float(V[c["livre"]]))
    for arch in par:
        for k in ("moins_moyenne", "moins_meilleur", "moins_livre"):
            out["gain_sur_ses_reseaux"][f"{arch}.{k}"] = t_apparie([par[arch][s][k] for s in GRAINES if s in par[arch]])
    for k in ("moins_moyenne", "moins_meilleur", "moins_livre"):
        out["gain_sur_ses_reseaux"][f"cnn_moins_mlp.{k}"] = t_apparie(
            [par["cnn"][s][k] - par["mlp"][s][k] for s in GRAINES if s in par.get("cnn", {}) and s in par.get("mlp", {})])
    # 3. la décomposition du score de Brier de l'ensemble des 32
    for k in ("brier_moy", "amb", "brier_ens", "desacc", "amb_chat"):
        d = []
        for s in GRAINES:
            a, b = D.get(f"cnn_INDEPC_s{s}"), R45.get(f"mlp_INDEPC_s{s}")
            if a is None or b is None:
                continue
            d.append(a["fin_choix"]["regles"]["TOUS"]["diversite"][k] - b["fin_choix"]["regles"]["TOUS"]["diversite"][k])
        out["brier"][f"cnn_moins_mlp.{k}"] = t_apparie(d)
        for arch, src in (("cnn", D), ("mlp", R45)):
            xs = [src[f"{arch}_INDEPC_s{s}"]["fin_choix"]["regles"]["TOUS"]["diversite"][k] for s in GRAINES
                  if f"{arch}_INDEPC_s{s}" in src]
            out["brier"][f"{arch}.{k}"] = float(np.mean(xs)) if xs else None
    json.dump(out, open(os.path.join(ICI, "apres46.json"), "w"), ensure_ascii=False, indent=1)
    pt = lambda v: "—" if v is None else f"{100 * v['ecart']:+.3f} [{100 * v['ic95'][0]:+.3f} ; {100 * v['ic95'][1]:+.3f}]"
    for nom, e in out["decomposition"].items():
        print(f"{nom:<22} écart {pt(e['ecart'])}   chat {pt(e['part_chat'])}   autres {pt(e['part_autres'])}")
    for k, v in out["gain_sur_ses_reseaux"].items():
        print(f"{k:<28} {pt(v)}")
    for k, v in out["brier"].items():
        print(f"{k:<24} {v if not isinstance(v, dict) else (v['ecart'], v['ic95'])}")
