"""Analyse de la clé 44 (descriptive, et comparaisons sans prédiction, signalées comme telles) :
python3 ana44.py [--references DOSSIER] → ana44.json

1. Par bras (graines 100 à 147 ; ENS, ENSR, INDEP, SYS, M32 de cette clé ; GRAND, SEUL, COURT2, INDEP0 repris de la
   clé 43) : les mesures de verdict44 (V_cal, V_fin, V_max, H ; E_cal, E_fin, E_max, E_vie, H_ens) ; la diversité
   finale (ambiguïté, sur les chats, désaccord, Brier moyen et de l'ensemble) ; la diversité et les lignées au fil des
   passages du juge ; pour ENS et ENSR, les apports à l'ensemble (minimum, maximum, celui du réseau livré) et la
   dispersion des réglages au fil des générations ; les durées.
2. Comparaisons sans prédiction : chaque ensemble (ENS, ENSR, INDEP, SYS, M32) contre les autres et contre GRAND,
   COURT2, SEUL, INDEP0, sur la vérité à seuil réglé, au seuil par défaut et au meilleur seuil ; les réseaux livrés
   entre eux ; la diversité entre bras ; l'ensemble à rappel du chat égal (0,85, 0,90, 0,92).
3. Le lien entre diversité et gain de l'ensemble sur son réseau livré : corrélation de Pearson, par bras sur les
   graines, et entre bras sur les moyennes.
4. L'ensemble avant et après la dernière sélection (au seuil par défaut : passage 119 du juge, et mesures finales).
Écrit après la rédaction du pré-enregistrement, avant toute course du plan : n'a rien affiché des données de la clé 44.
"""
import glob, json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
from verdict44 import mesures, GRAINES, GRILLE, EMPREINTES43

NOUVEAUX = ("ENS", "ENSR", "INDEP", "SYS", "M32")
REPRIS = ("GRAND", "SEUL", "COURT2", "INDEP0")
ENSEMBLES = ("ENS", "ENSR", "INDEP", "SYS", "M32")


def charger_brut(dref):
    D = {}
    for f in sorted(glob.glob(os.path.join(ICI, "resultats", "tranche_E*.json"))):
        D.update(json.load(open(f)))
    R = {}
    for motif in ("tranche_E*.json", "tranche_L*.json", "tranche_C*.json"):
        for f in sorted(glob.glob(os.path.join(dref, motif))):
            R.update(json.load(open(f)))
    for b in REPRIS:
        for s in GRAINES:
            k = f"mlp_{b}_s{s}"
            if k in R:
                D[k] = R[k]
    return D


def t_apparie(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("nan")
    q = stats.t.ppf(0.975, len(d) - 1)
    return dict(n=len(d), ecart=float(d.mean()), t=float(t), p_bilateral=float(2 * stats.t.sf(abs(t), len(d) - 1)) if se > 0 else None,
                ic95=[float(d.mean() - q * se), float(d.mean() + q * se)], gagne=int((d > 0).sum()))


def ao_a(front, cible):
    Rf, Ao = np.array(front["Rf"]), np.array(front["Ao"])
    o = np.argsort(Rf, kind="stable")
    return float(np.interp(cible, Rf[o], Ao[o])) if Rf.min() <= cible <= Rf.max() else None


def resume(D, M, b):
    noms = [f"mlp_{b}_s{s}" for s in GRAINES if f"mlp_{b}_s{s}" in D]
    if not noms:
        return None
    rs, ms = [D[k] for k in noms], [M[k] for k in noms]
    e = dict(n=len(rs))
    for c in ("V_cal", "V_fin", "V_max", "V_vie", "H", "E_cal", "E_fin", "E_max", "E_vie", "H_ens", "AMB", "DES", "LIG",
              "b_cal", "eb_cal"):
        if all(c in x for x in ms):
            v = np.array([x[c] for x in ms], float)
            e[c] = float(v.mean()); e[c + "_sd"] = float(v.std(ddof=1)); e[c + "_med"] = float(np.median(v))
    if all("fin_ens" in r and "diversite" in r["fin_ens"] for r in rs):
        for c in ("amb", "amb_chat", "desacc", "brier_moy", "brier_ens"):
            e["fin_" + c] = float(np.mean([r["fin_ens"]["diversite"][c] for r in rs]))
    js = [{d["g"]: d for d in r["juge"]} for r in rs]
    if all("amb" in j[0] for j in js):
        e["vie_amb"] = {g: float(np.mean([j[g]["amb"] for j in js])) for g in GRILLE}
        e["vie_desacc"] = {g: float(np.mean([j[g]["desacc"] for j in js])) for g in GRILLE}
    if all("lignees" in j[119] for j in js):
        e["vie_lignees"] = {g: float(np.mean([j[g]["lignees"] for j in js])) for g in GRILLE}
    if all("V30_ens" in j[119] for j in js):
        e["vie_E"] = {g: float(np.mean([j[g]["V30_ens"] for j in js])) for g in GRILLE}
        e["ens_avant_apres_derniere_selection"] = t_apparie(
            [r["fin_ens"]["V30_ens"] - j[119]["V30_ens"] for r, j in zip(rs, js)])
    if all("apport_min" in r["tr"][0] for r in rs):
        for c in ("apport_min", "apport_max", "apport_livre", "lv_ens", "llr_sd", "lwd_sd"):
            e["tr_" + c] = {g: float(np.mean([r["tr"][g][c] for r in rs])) for g in (0, 8, 40, 80, 119)}
        e["generations_livre_protege"] = float(np.mean([sum(d["livre_protege"] for d in r["tr"]) for r in rs]))
    if all("fin_ens" in r for r in rs):
        e["ens_Ao_a"] = {c: float(np.mean([ao_a(r["fin_ens"]["front"], c) for r in rs])) for c in (0.85, 0.90, 0.92)}
    e["livre_Ao_a"] = {c: float(np.mean([ao_a(r["fin_livre"], c) for r in rs])) for c in (0.85, 0.90, 0.92)}
    e["duree_min"] = float(np.mean([r["duree"] for r in rs]) / 60) if all("duree" in r for r in rs) else None
    return e


def comparaisons(M, D):
    out = {}

    def serie(b, m):
        return {s: M[f"mlp_{b}_s{s}"][m] for s in GRAINES if f"mlp_{b}_s{s}" in M and m in M[f"mlp_{b}_s{s}"]}

    def paire(A, B):
        x, y = serie(*A), serie(*B)
        g = sorted(set(x) & set(y))
        if len(g) >= 3:
            out[f"{A[0]}.{A[1]}_moins_{B[0]}.{B[1]}"] = t_apparie([x[s] - y[s] for s in g])

    for i, a in enumerate(ENSEMBLES):
        for b in ENSEMBLES[i + 1:]:
            for m in ("E_cal", "E_fin", "E_max", "AMB", "DES", "LIG", "V_cal"):
                paire((a, m), (b, m))
        for ref in ("GRAND", "COURT2", "SEUL"):
            for (ma, mb) in (("E_cal", "V_cal"), ("E_fin", "V_fin"), ("E_max", "V_max")):
                paire((a, ma), (ref, mb))
        paire((a, "E_cal"), ("INDEP0", "E_cal"))
        paire((a, "E_cal"), (a, "V_cal"))
    for a in ENSEMBLES:
        for ref in ("GRAND", "COURT2"):
            for c in (0.85, 0.90, 0.92):
                d = []
                for s in GRAINES:
                    ka, kb = f"mlp_{a}_s{s}", f"mlp_{ref}_s{s}"
                    if ka in D and kb in D:
                        x, y = ao_a(D[ka]["fin_ens"]["front"], c), ao_a(D[kb]["fin_livre"], c)
                        if x is not None and y is not None:
                            d.append(x - y)
                if len(d) >= 3:
                    out[f"{a}.ens_moins_{ref}_Ao_{c:.2f}"] = t_apparie(d)
    return out


def liens(M):
    """La diversité finale (AMB) et le gain de l'ensemble sur son réseau livré (E_cal − V_cal)."""
    out = {}
    moy = []
    for b in ENSEMBLES:
        xs = [(M[f"mlp_{b}_s{s}"]["AMB"], M[f"mlp_{b}_s{s}"]["E_cal"] - M[f"mlp_{b}_s{s}"]["V_cal"])
              for s in GRAINES if f"mlp_{b}_s{s}" in M and "AMB" in M[f"mlp_{b}_s{s}"]]
        if len(xs) >= 3:
            a, g = np.array(xs).T
            r, p = stats.pearsonr(a, g)
            out[f"{b}.par_graine"] = dict(r=float(r), p=float(p), n=len(xs))
            moy.append((float(a.mean()), float(g.mean())))
    if len(moy) >= 3:
        a, g = np.array(moy).T
        out["entre_bras"] = dict(r=float(np.corrcoef(a, g)[0, 1]), n=len(moy), points=moy)
    return out


if __name__ == "__main__":
    dref = sys.argv[sys.argv.index("--references") + 1] if "--references" in sys.argv else \
        os.path.join(ICI, "..", "cle43", "resultats")
    D = charger_brut(dref)
    M = {k: mesures(v) for k, v in D.items()}
    out = dict(bras={b: resume(D, M, b) for b in NOUVEAUX + REPRIS}, comparaisons=comparaisons(M, D), liens=liens(M))
    json.dump(out, open(os.path.join(ICI, "ana44.json"), "w"), ensure_ascii=False)
    for b, e in out["bras"].items():
        if e is None:
            print(f"{b:<7} absent"); continue
        ligne = f"{b:<7} n={e['n']:>2} V_cal {e['V_cal']:.4f} V_fin {e['V_fin']:.4f} H {100 * e['H']:+.2f}"
        if "E_cal" in e:
            ligne += f" | E_cal {e['E_cal']:.4f} E_fin {e['E_fin']:.4f} E_max {e['E_max']:.4f} H_ens {100 * e['H_ens']:+.2f}"
        if "AMB" in e:
            ligne += f" | AMB {e['AMB']:.4f} DES {e['DES']:.3f} LIG {e.get('LIG', float('nan')):.1f}"
        print(ligne + (f"  {e['duree_min']:.1f} min" if e.get("duree_min") else ""))
    for k, v in out["comparaisons"].items():
        print(f"{k:<40} {100 * v['ecart']:+7.2f}  t = {v['t']:+6.2f}  IC [{100 * v['ic95'][0]:+.2f} ; "
              f"{100 * v['ic95'][1]:+.2f}]  gagne {v['gagne']}/{v['n']}")
    for k, v in out["liens"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "points"})
