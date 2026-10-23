"""Les essais du moteur de la clé 47, avant le plan : python3 essai47.py [--vrai]

Sans --vrai (petit modèle du calcul : 2 pas par génération au lieu de 100, un juge de 2 500 images, 4 générations) :
  1. pour chaque bras du plan, une course d'une traite, puis la même coupée (après une sauvegarde donnée) et reprise
     depuis son point : les deux résultats doivent être identiques, au bit près (leurs JSON, caractère par caractère) ;
  2. les identités : K1 (la recette coupée en un réseau), le réseau du VEILLEUR et le parent de la SOUPE doivent refaire
     au bit près la course SEUL10 du moteur de la clé 46 (course_seul), sur ce qu'ils ont en commun.
Avec --vrai : les identités de 2, au vrai calcul (100 pas par génération, le juge entier), sur 2 générations, et la
mesure des durées (secondes par génération d'entraînement, pour chaque bras).
Graines hors plan (300 et suivantes). Écrit essai47_bilan.json.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, tempfile, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import numpy as np
import echelle47 as E
import reprise47 as R

VRAI = "--vrai" in sys.argv
LR10 = E.LR0 * 0.0625


def petit():
    E.PAS = 2
    Xtr, ytr, Xte, yte = E.donnees()
    E._D["brut"] = (Xtr, ytr, Xte[:2500], yte[:2500])
    E._D.pop("juge_propre", None)


def js(r):
    return json.dumps(r, sort_keys=True)


def coupe_reprise(f, arrets):
    """f(point, arret) : une course. Rend (d'une traite, [coupée à n puis reprise, pour n dans arrets])."""
    ref = f(None, None)
    out = []
    for n in arrets:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "point.pkl")
            try:
                f(p, n)
                coupee = False
            except (E.Arret, R.Arret):
                coupee = True
            r = f(p, None)
            out.append(dict(arret=n, coupee=coupee, identique=js(r) == js(ref)))
    return ref, out


def commun(a, b, cles=("tr", "juge")):
    """Les valeurs communes de deux courses (mêmes clés, mêmes positions) : (nombre comparé, nombre d'écarts)."""
    n = e = 0
    for c in cles:
        for x, y in zip(a[c], b[c]):
            for k in set(x) & set(y):
                n += 1
                e += x[k] != y[k]
    return n, e


def main():
    bilan = {}
    if not VRAI:
        petit()
        G = 4
        cas = {
            "K2": lambda p, n: R.course("K2", "cnn", 301, gens=G, point=p, arret=n),
            "K4P": lambda p, n: R.course("K4P", "cnn", 302, gens=G, point=p, arret=n),
            "K8": lambda p, n: R.course("K8", "cnn", 303, gens=G, point=p, arret=n),
            "K32": lambda p, n: R.course("K32", "cnn", 304, gens=G, point=p, arret=n),
            "POPC": lambda p, n: R.course("POPC", "cnn", 305, gens=G, point=p, arret=n),
            "VEILLEUR": lambda p, n: R.course("VEILLEUR", "cnn", 306, gens=G, point=p, arret=n),
            "TOILE": lambda p, n: R.course("TOILE", "cnn", 307, gens=G, point=p, arret=n),
            "SOUPE": lambda p, n: R.course("SOUPE", "cnn", 308, gens=G, point=p, arret=n),
        }
        # où couper : après la 2e sauvegarde (une génération faite), et au milieu des mesures finales ;
        # K4P : aussi pendant la distillation ; SOUPE : juste après la naissance des enfants ; POPC : générations
        arrets = {"K2": [2, 6], "K4P": [2, 6, 9, 13], "K8": [2, 6], "K32": [2, 6], "POPC": [1], "VEILLEUR": [2, 5],
                  "TOILE": [2, 6], "SOUPE": [2, 3, 6]}
        refs = {}
        for b, f in cas.items():
            t0 = time.time()
            refs[b], res = coupe_reprise(f, arrets[b])
            bilan[f"reprise_{b}"] = res
            print(b, res, f"{time.time() - t0:.0f} s", flush=True)
        g_ident = G
    else:
        g_ident = 2
    # les identités avec la course SEUL10 du moteur de la clé 46
    t0 = time.time()
    s10 = E.course_seul("cnn", 310, 10, E.FACTEUR, False, g_ident, True, LR10)
    d_s10 = time.time() - t0
    k1 = E.course_coupe("cnn", 310, 1, 10, g_ident, True)
    v = E.course_veilleur("cnn", 310, 10, g_ident, True)
    so = E.course_soupe("cnn", 310, 10, E.ENFANTS, g_ident, True)
    gm = g_ident // 2
    so_parent = dict(tr=so["tr"][:gm], juge=[j for j in so["juge"] if j["g"] < gm])
    s10_moitie = dict(tr=s10["tr"][:gm], juge=[j for j in s10["juge"] if j["g"] < gm])
    bilan["identite_K1"] = dict(trajet=commun(k1, s10), fin_livre=js(k1["fin_livre"]) == js(s10["fin_livre"]))
    bilan["identite_VEILLEUR"] = dict(trajet=commun(v, s10), fin_livre=js(v["fin_livre"]) == js(s10["fin_livre"]))
    bilan["identite_SOUPE_parent"] = dict(trajet=commun(so_parent, s10_moitie), generations=gm)
    print("identités :", {k: bilan[k] for k in bilan if k.startswith("identite")}, flush=True)
    if VRAI:
        # les durées : secondes par génération d'entraînement (sans le juge), pour chaque bras
        dur = {"SEUL10 (course_seul, 2 générations, avec mesures)": d_s10}
        for nom, k in (("K1", 1), ("K2", 2), ("K4", 4), ("K8", 8), ("K32", 32)):
            with E.taille(k):
                rng = np.random.default_rng(0)
                pop = E.Seul(rng, "cnn", E.C_ANCRE, 10)
                W = E.Monde(399, "fixe")
                TX, Ty, TI = E.arrivees_idx(W, E.TAMPON, 0)
                n = max(2, 64 // k)
                t0 = time.time()
                for _ in range(n):
                    idx = rng.integers(0, E.TAMPON, (k, E.NB))
                    pop.pas_seul(TX[idx], Ty[idx], 0.01, E.WD0)
                dt = (time.time() - t0) / n
                dur[nom] = dict(s_par_pas=dt, s_par_generation=dt * E.PAS * E.FACTEUR // k)
        rng = np.random.default_rng(0)
        pop = E.Population(rng, "cnn", "BAS", E.C_ANCRE, 10, genome=E.GENOME_CNN)
        pop.sans_curation = True; pop.eps[:] = 0.0
        W = E.Monde(399, "fixe")
        TX, Ty = W.arrivees(E.TAMPON, 0)
        t0 = time.time()
        for _ in range(4):
            idx = pop.tirer(E.NB)
            pop.pas(TX[idx], Ty[idx], idx)
        dt = (time.time() - t0) / 4
        dur["POPC"] = dict(s_par_pas=dt, s_par_generation=dt * E.PAS)
        bilan["durees"] = dur
        print(json.dumps(dur, indent=1), flush=True)
    nom = "essai47_bilan_vrai.json" if VRAI else "essai47_bilan.json"
    json.dump(bilan, open(os.path.join(ICI, nom), "w"), indent=1)
    ok = all(all(x["identique"] and x["coupee"] for x in v) for k, v in bilan.items() if k.startswith("reprise_"))
    ok = ok and all(v["trajet"][1] == 0 and v.get("fin_livre", True) for k, v in bilan.items() if k.startswith("identite"))
    print("BILAN :", "bon" if ok else "ÉCART", flush=True)


if __name__ == "__main__":
    main()
