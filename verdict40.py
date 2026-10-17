"""Le verdict de la clé 40, selon la règle écrite dans PREENREGISTREMENT_40.md : python3 verdict40.py

Lit resultats/tranche_P*.json et resultats/tranche_R*.json (les 192 courses du plan, en 32×32) et
reference39.json (les mêmes mesures pour les courses publiées de la clé 39, en 16×16), calcule pour chaque
course les mesures pré-enregistrées, applique les 19 tests (4 principaux, corrigés ensemble par la méthode de
Holm ; 15 secondaires) et écrit verdicts40.json.

Mesures (le « réseau livré » : le mieux noté par l'ancre pour SYS et INDEP, le réseau lui-même sinon) :
  V_fin  la vérité ×30 du juge pour le réseau livré, à la génération 119 (V30_b) ;
  V_vie  sa moyenne sur les 16 passages 0, 8, …, 112, 119 ;
  H      le biais d'auto-évaluation : note de l'ancre moins vérité du juge, pour le réseau livré, en moyenne
         sur les 11 passages 40, 48, …, 112, 119 ;
  E_fin  la vérité ×30 de l'ensemble des 32 réseaux (moyenne des probabilités) à la génération 119.
Écart d'un test, par graine : A − B dans la clé 40 ; pour un test d'échelle, (A − B) en 32×32 moins (A − B) en
16×16 à la même graine (la même suite d'images) ; pour un test d'échelle simple, A en 32×32 moins A en 16×16.
Règle : Student apparié par graine ; p unilatéral exact (n − 1 degrés de liberté).
  principaux : confirmée si la méthode de Holm, au niveau 0,025 unilatéral sur les 4, la retient ;
               démentie si le p dans le sens contraire est sous 0,025 / 4 ; indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  équivalence (S14) : deux tests unilatéraux, bornes ±0,01, chacun à 0,025 ; démentie si l'écart est
               significativement au-delà d'une borne (p < 0,025) ; indécidable sinon.
  Plus de deux graines manquantes pour un bras : le test est « incomplet ».
Lecture pré-enregistrée (rapportée sans test supplémentaire) : « gain de capacité de la population » sur une
architecture si le système bat chacune de ses recettes standard (SEUL, SEUL10, GRAND, COURT pour le convolutif ;
SEUL, GRAND, COURT pour le perceptron), chacune à p < 0,025 unilatéral.
"""
import glob, json, math, os
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
TARDIF = [g for g in GRILLE if g >= 40]
GRAINES = {"cnn": range(12), "mlp": range(24)}
ALPHA, BORNE = 0.025, 0.01
STANDARD = {"cnn": ("SEUL", "SEUL10", "GRAND", "COURT"), "mlp": ("SEUL", "GRAND", "COURT")}

# (identifiant, famille, architecture, (bras A, mesure A), (bras B, mesure B), sens attendu, échelle, énoncé)
TESTS = [
    ("P1", "principal", "cnn", ("GRAND", "V_fin"), ("SYS", "V_fin"), +1, None,
     "convolutif : en 32×32, le grand réseau bat le système entier"),
    ("P2", "principal", "cnn", ("GRAND", "V_fin"), ("SYS", "V_fin"), +1, "ecart",
     "convolutif : l'avance du grand réseau sur le système grandit de 16×16 à 32×32"),
    ("P3", "principal", "mlp", ("SYS", "V_fin"), ("GRAND", "V_fin"), +1, None,
     "perceptron : en 32×32, le système entier bat encore le grand réseau"),
    ("P4", "principal", "mlp", ("SYS", "V_fin"), ("SEUL", "V_fin"), +1, None,
     "perceptron : en 32×32, le système entier bat le réseau seul qui reçoit tout le calcul"),
    ("S1", "secondaire", "cnn", ("SYS", "V_fin"), ("INDEP", "V_fin"), +1, None,
     "convolutif : la sélection bat la recherche aléatoire"),
    ("S2", "secondaire", "mlp", ("SYS", "V_fin"), ("INDEP", "V_fin"), +1, None,
     "perceptron : la sélection bat la recherche aléatoire"),
    ("S3", "secondaire", "cnn", ("SYS", "V_fin"), ("COURT", "V_fin"), +1, None,
     "convolutif : la population vaut ses 32 réseaux"),
    ("S4", "secondaire", "mlp", ("SYS", "V_fin"), ("COURT", "V_fin"), +1, None,
     "perceptron : la population vaut ses 32 réseaux"),
    ("S5", "secondaire", "cnn", ("SEUL10", "V_fin"), ("SEUL", "V_fin"), +1, None,
     "convolutif : la bascule à 10 améliore la recette standard"),
    ("S6", "secondaire", "cnn", ("SYS", "V_fin"), ("SEUL", "V_fin"), +1, None,
     "convolutif : le système entier bat le réseau seul qui reçoit tout le calcul"),
    ("S7", "secondaire", "cnn", ("GRAND", "V_fin"), ("SEUL", "V_fin"), +1, None,
     "convolutif : à calcul égal, le grand réseau bat le petit réseau entraîné longtemps"),
    ("S8", "secondaire", "cnn", ("SEUL", "V_vie"), ("SYS", "V_vie"), +1, None,
     "convolutif : la recette standard protège mieux pendant l'apprentissage"),
    ("S9", "secondaire", "mlp", ("SYS", "V_vie"), ("SEUL", "V_vie"), +1, None,
     "perceptron : le système protège mieux que le réseau seul pendant l'apprentissage"),
    ("S10", "secondaire", "cnn", ("SYS", "H"), ("SEUL", "H"), +1, None,
     "convolutif : le système se surestime plus que le réseau seul"),
    ("S11", "secondaire", "mlp", ("SYS", "H"), ("SEUL", "H"), +1, None,
     "perceptron : le système se surestime plus que le réseau seul"),
    ("S12", "secondaire", "cnn", ("INDEP", "E_fin"), ("INDEP", "V_fin"), +1, None,
     "convolutif : l'ensemble des 32 indépendants bat le meilleur d'entre eux"),
    ("S13", "secondaire", "mlp", ("INDEP", "E_fin"), ("INDEP", "V_fin"), +1, None,
     "perceptron : l'ensemble des 32 indépendants bat le meilleur d'entre eux"),
    ("S14", "equivalence", "cnn", ("SYS", "E_fin"), ("SYS", "V_fin"), 0, None,
     "convolutif : l'ensemble du système n'ajoute pas plus d'un point à son réseau livré"),
    ("S15", "secondaire", "cnn", ("SYS", "V_fin"), ("SYS", "V_fin"), +1, "simple",
     "convolutif : en 32×32, le système entier est plus vrai qu'en 16×16"),
]


def charger():
    D = {}
    for f in sorted(glob.glob(os.path.join(ICI, "resultats", "tranche_P*.json")) +
                    glob.glob(os.path.join(ICI, "resultats", "tranche_R*.json"))):
        for k, v in json.load(open(f)).items():
            if k in D:
                assert json.dumps(D[k]["juge"]) == json.dumps(v["juge"]), f"course en double, différente : {k}"
            D[k] = v
    return D


def mesures(r):
    j = {d["g"]: d for d in r["juge"]}
    out = dict(V_fin=j[119]["V30_b"], V_vie=float(np.mean([j[g]["V30_b"] for g in GRILLE])),
               H=float(np.mean([j[g]["note_b"] - j[g]["V30_b"] for g in TARDIF])))
    if "V30_ens" in j[119]:
        out["E_fin"] = j[119]["V30_ens"]
    return out


def serie(D, arch, bras, m):
    return {s: mesures(D[f"{arch}_{bras}_s{s}"])[m] for s in GRAINES[arch] if f"{arch}_{bras}_s{s}" in D}


def serie39(R, arch, bras, m):
    return {s: R[f"{arch}_{bras}_s{s}"][m] for s in GRAINES[arch] if f"{arch}_{bras}_s{s}" in R}


def student(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    t = d.mean() / se if se > 0 else math.copysign(math.inf, d.mean())
    return n, float(d.mean()), float(d.std(ddof=1)), float(t), float(stats.t.sf(t, n - 1)), float(stats.t.cdf(t, n - 1))


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


def ecarts(D, R, arch, A, B, echelle):
    """Les écarts par graine d'un test, et le nombre de graines manquantes."""
    a40, b40 = serie(D, arch, *A), serie(D, arch, *B)
    if echelle is None:
        g = sorted(set(a40) & set(b40))
        d = {s: a40[s] - b40[s] for s in g}
    elif echelle == "ecart":
        a39, b39 = serie39(R, arch, *A), serie39(R, arch, *B)
        g = sorted(set(a40) & set(b40) & set(a39) & set(b39))
        d = {s: (a40[s] - b40[s]) - (a39[s] - b39[s]) for s in g}
    else:
        a39 = serie39(R, arch, *A)
        g = sorted(set(a40) & set(a39))
        d = {s: a40[s] - a39[s] for s in g}
    manque = max(len(GRAINES[arch]) - len(a40), len(GRAINES[arch]) - len(b40))
    return d, manque


if __name__ == "__main__":
    D = charger()
    R = json.load(open(os.path.join(ICI, "reference39.json")))
    res = []
    for tid, fam, arch, A, B, sens, echelle, texte in TESTS:
        d, manque = ecarts(D, R, arch, A, B, echelle)
        e = dict(id=tid, famille=fam, arch=arch, A=f"{A[0]}.{A[1]}", B=f"{B[0]}.{B[1]}", echelle=echelle,
                 enonce=texte, manquantes=manque)
        if manque > 2 or len(d) < 3:
            e.update(verdict="incomplet"); res.append(e); continue
        n, moy, sd, t, p_haut, p_bas = student(list(d.values()))
        se = sd / math.sqrt(n)
        q = float(stats.t.ppf(0.975, n - 1))
        e.update(n=n, ecart=moy, sd=sd, t=t, ic95=[moy - q * se, moy + q * se])
        if fam == "equivalence":
            t_bas, t_haut = (moy + BORNE) / se, (moy - BORNE) / se
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
        e["p_holm_19"] = a
    # la lecture pré-enregistrée : le système bat-il chacune des recettes standard ?
    lecture = {}
    for arch in ("cnn", "mlp"):
        lignes = {}
        for b in STANDARD[arch]:
            d, manque = ecarts(D, R, arch, ("SYS", "V_fin"), (b, "V_fin"), None)
            if manque > 2 or len(d) < 3:
                lignes[b] = None; continue
            n, moy, sd, t, p_haut, _ = student(list(d.values()))
            lignes[b] = dict(ecart=moy, t=t, p=p_haut)
        gain = all(v is not None and v["p"] < ALPHA for v in lignes.values())
        lecture[arch] = dict(contre=lignes, gain_de_capacite=gain)
    json.dump(dict(tests=res, lecture=lecture), open(os.path.join(ICI, "verdicts40.json"), "w"),
              ensure_ascii=False, indent=1)
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} {e['arch']}  incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        print(f"{e['id']:<4} {e['arch']}  {e['verdict']:<11} écart {100 * e['ecart']:+6.2f} pt  t = {e['t']:+6.2f}"
              f"  p = {e['p']:.2g}  (Holm sur 19 : {e['p_holm_19']:.2g})  {e['enonce']}")
    for arch, v in lecture.items():
        print(f"gain de capacité de la population ({arch}) : {'oui' if v['gain_de_capacite'] else 'non'}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
