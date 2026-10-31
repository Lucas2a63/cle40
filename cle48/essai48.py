"""Les essais du moteur de la clé 48, avant le plan : python3 essai48.py [--vrai]

Sans --vrai (petit modèle du calcul : 2 pas par génération au lieu de 100, un juge de 2 500 images, 4 générations — 8
pour TOILE8) :
  1. pour chacun des onze bras du plan, une course d'une traite, puis la même coupée (après une sauvegarde donnée de
     son point de reprise) et reprise depuis son point : les deux résultats doivent être identiques, au bit près (leurs
     JSON, caractère par caractère) ;
  2. les identités, sur tout ce que les deux courses ont en commun :
     - course_coupe48 avec div = 1 refait course_coupe de la clé 47 (K2 ; K4P avec l'élève, eleve = 1) ;
     - course_toile48 à quatre cycles refait course_toile de la clé 47 (TOILE) ;
     - un réseau seul de course_coupe48 (k = 1) refait course_seul de la clé 46 au facteur FACTEUR / div et au taux
       LR0 · lr_coupe(div) : SEUL10 (div = 1), SEUL_H (2), SEUL_Q (4), SEUL_S (16) ;
     - l'élève de PHOTO2 et SEUL_H partent des mêmes poids et tirent les mêmes paquets (générateurs dans le même état
       à la fin), dans le même monde (mêmes recyclages, mêmes ancres) ;
  3. au petit modèle encore, sur 12 générations (la bascule ×30 de la 10e génération comprise) : SEUL_S contre
     course_seul (clé 46), K2 (div = 1) contre course_coupe et la Toile à quatre cycles de trois générations contre
     course_toile (clé 47) ;
  4. l'empreinte du monde : six bras sur une même graine (SEUL10, SEUL_S, K2_S, K4_S, TOILE4, PHOTO2 et son élève) ont
     la même ; une autre graine en a une autre.
Avec --vrai : les identités de 2 (sauf K4P ; la Toile à deux cycles), au vrai calcul (100 pas par génération, le juge
entier), sur 2 générations, et les durées dans le bac à sable : secondes par pas de gradient pour chaque façon de
couper (un réseau, 2 ou 4 réseaux à la fois), et durée des mesures finales d'une course courte.
Graines hors plan (320 et suivantes). Écrit essai48_bilan.json (ou essai48_bilan_vrai.json).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, pickle, sys, tempfile, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import numpy as np
import echelle48 as E
import reprise48 as R

VRAI = "--vrai" in sys.argv


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
            except E.Arret:
                coupee = True
            r = f(p, None)
            out.append(dict(arret=n, coupee=coupee, identique=js(r) == js(ref)))
    return ref, out


def en_commun(a, b, sauf=("calcul", "photo")):
    """Compare deux résultats sur leurs clés communes (hors sauf), et « calcul » sur ses clés communes : (nombre de
    clés comparées, liste des clés qui diffèrent)."""
    cles = sorted((set(a) & set(b)) - set(sauf))
    diff = [k for k in cles if js(a[k]) != js(b[k])]
    if "calcul" in a and "calcul" in b:
        cc = sorted(set(a["calcul"]) & set(b["calcul"]))
        cles.append("calcul[" + ",".join(cc) + "]")
        if any(a["calcul"][c] != b["calcul"][c] for c in cc):
            diff.append("calcul")
    return len(cles), diff


def trajet(a, b):
    """Les valeurs communes du trajet et du juge de deux courses : (nombre comparé, nombre d'écarts)."""
    n = e = 0
    for c in ("tr", "juge"):
        assert len(a[c]) == len(b[c])
        for x, y in zip(a[c], b[c]):
            for k in set(x) & set(y):
                n += 1
                e += x[k] != y[k]
    return n, e


def identites(G):
    bilan = {}
    t0 = time.time()
    a, b = E.course_coupe("cnn", 321, 2, 10, G, True), E.course_coupe48("cnn", 321, 2, 1, 10, G, True)
    bilan["K2_47_contre_coupe48"] = en_commun(a, b)
    if not VRAI:
        a = E.course_coupe("cnn", 322, 4, 10, G, True, True)
        b = E.course_coupe48("cnn", 322, 4, 1, 10, G, True, eleve=1)
        bilan["K4P_47_contre_coupe48"] = dict(maitre=en_commun(a, b), eleve=en_commun(a["photo"], b["photo"]))
    cy = 4 if G % 4 == 0 else 2                          # au vrai calcul, sur 2 générations : deux cycles
    a, b = E.course_toile("cnn", 323, 10, cy, G, True), E.course_toile48("cnn", 323, 10, cy, G, True)
    bilan["TOILE_47_contre_toile48"] = en_commun(a, b)
    for nom, div in (("SEUL10", 1), ("SEUL_H", 2), ("SEUL_Q", 4), ("SEUL_S", 16)):
        s = E.course_seul("cnn", 324, 10, E.FACTEUR // div, False, G, True, E.LR0 * E.lr_coupe(div))
        c = R.course(nom, "cnn", 324, gens=G)
        bilan[f"{nom}_contre_course_seul46"] = dict(trajet=trajet(s, c), fin_livre=js(s["fin_livre"]) == js(c["fin_livre"]),
                                                    recycle=s["recycle"] == c["recycle"])
    # l'élève de PHOTO2 et SEUL_H : mêmes poids de départ, mêmes paquets, même monde
    W = E.Monde(325, "fixe")
    with E.taille(1):
        _, p1, *_ = E._depart(W, 325, "cnn", 10)
        W = E.Monde(325, "fixe")
        _, p2, *_ = E._depart(W, 325, "cnn", 10, E.SeulPhoto)
    depart = all(np.array_equal(getattr(p1.net, k), getattr(p2.net, k)) for k in p1.net.POIDS)
    with tempfile.TemporaryDirectory() as d:
        ph, sh = os.path.join(d, "p.pkl"), os.path.join(d, "s.pkl")
        rp = R.course("PHOTO2", "cnn", 325, gens=G, point=ph)
        rs = R.course("SEUL_H", "cnn", 325, gens=G, point=sh)
        Pp, Ps = pickle.load(open(ph, "rb")), pickle.load(open(sh, "rb"))
        etat = (js(Pp["photo"]["rng"].bit_generator.state) == js(Ps["coupe"]["rng"].bit_generator.state))
        monde = (rp["photo"]["recycle"] == rs["recycle"]
                 and all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2])
                         for x, y in zip(Pp["photo"]["ancres"], Ps["coupe"]["ancres"])))
    bilan["eleve_PHOTO2_et_SEUL_H"] = dict(memes_poids_de_depart=bool(depart), generateurs_au_meme_etat=bool(etat),
                                          meme_monde=bool(monde),
                                          trajets_differents=js(rp["photo"]["tr"]) != js(rs["tr"]))
    print("identités :", json.dumps(bilan), f"({time.time() - t0:.0f} s)", flush=True)
    return bilan


def douze_generations():
    """Les identités sur 12 générations (la bascule ×30 de la 10e génération comprise), au petit modèle."""
    G = 12
    out = {}
    s = E.course_seul("cnn", 326, 10, E.FACTEUR // 16, False, G, True, E.LR0 * E.lr_coupe(16))
    c = R.course("SEUL_S", "cnn", 326, gens=G)
    out["SEUL_S_contre_course_seul46_12g"] = dict(trajet=trajet(s, c), fin_livre=js(s["fin_livre"]) == js(c["fin_livre"]),
                                                  recycle=s["recycle"] == c["recycle"])
    a, b = E.course_coupe("cnn", 327, 2, 10, G, True), E.course_coupe48("cnn", 327, 2, 1, 10, G, True)
    out["K2_47_contre_coupe48_12g"] = en_commun(a, b)
    a, b = E.course_toile("cnn", 328, 10, 4, G, True), E.course_toile48("cnn", 328, 10, 4, G, True)
    out["TOILE_47_contre_toile48_12g"] = en_commun(a, b)
    print("12 générations :", json.dumps(out), flush=True)
    return out


def empreintes_du_monde():
    """Six bras sur la graine 341 (et l'élève de PHOTO2) : la même empreinte du monde ; SEUL_S sur la graine 342 : une
    autre."""
    em = {}
    for b in ("SEUL10", "SEUL_S", "K2_S", "K4_S", "TOILE4", "PHOTO2"):
        r = R.course(b, "cnn", 341, gens=4)
        em[b] = r["monde"]
        if "photo" in r:
            em["PHOTO2.eleve"] = r["photo"]["monde"]
    autre = R.course("SEUL_S", "cnn", 342, gens=4)["monde"]
    out = dict(meme_empreinte=len(set(em.values())) == 1, empreintes=em, autre_graine_differente=autre not in em.values())
    print("empreintes du monde :", json.dumps(out), flush=True)
    return dict(empreinte_monde=out)


def bon_identites(b):
    ok = True
    for k, v in b.items():
        if k.endswith("contre_coupe48") or k.endswith("contre_toile48"):
            for x in (v.values() if isinstance(v, dict) else [v]):
                ok = ok and not x[1]
        elif k.endswith("course_seul46"):
            ok = ok and v["trajet"][1] == 0 and v["fin_livre"] and v["recycle"]
        elif k == "eleve_PHOTO2_et_SEUL_H":
            ok = ok and all(v.values())
        elif k.endswith("_12g"):
            ok = ok and (not v[1] if isinstance(v, (list, tuple)) else (v["trajet"][1] == 0 and v["fin_livre"]
                                                                         and v["recycle"]))
        elif k == "empreinte_monde":
            ok = ok and v["meme_empreinte"] and v["autre_graine_differente"]
    return ok


def main():
    bilan = {}
    if not VRAI:
        petit()
        G = 4
        graines = {b: 330 + i for i, b in enumerate(("SEUL10", "SEUL_H", "PHOTO2", "TOILE4", "TOILE8", "SEUL_Q", "K2_Q",
                                                     "K4_Q", "SEUL_S", "K2_S", "K4_S"))}
        # où couper (le numéro de la sauvegarde après laquelle la course s'arrête) : après la première génération,
        # au milieu des mesures finales, à la dernière sauvegarde ; PHOTO2 aussi après les probabilités du maître et
        # pendant la distillation ; TOILE8 (8 générations) au milieu des cycles
        arrets = {"SEUL10": [2, 5, 6], "SEUL_H": [2, 5], "PHOTO2": [2, 6, 9, 11, 14, 15], "TOILE4": [2, 7, 10],
                  "TOILE8": [3, 11, 14], "SEUL_Q": [2, 6], "K2_Q": [2, 6, 8], "K4_Q": [2, 7], "SEUL_S": [2, 5],
                  "K2_S": [2, 6], "K4_S": [2, 5, 8]}
        for b, s in graines.items():
            g = 8 if b == "TOILE8" else G
            t0 = time.time()
            _, res = coupe_reprise(lambda p, n, b=b, s=s, g=g: R.course(b, "cnn", s, gens=g, point=p, arret=n),
                                   arrets[b])
            bilan[f"reprise_{b}"] = res
            print(b, res, f"{time.time() - t0:.0f} s", flush=True)
        bilan.update(identites(G))
        bilan.update(douze_generations())
        bilan.update(empreintes_du_monde())
    else:
        bilan.update(identites(2))
        # les durées : secondes par pas de gradient (k réseaux à la fois), sans le juge
        dur = {}
        for k in (1, 2, 4):
            with E.taille(k):
                rng = np.random.default_rng(0)
                pop = E.Seul(rng, "cnn", E.C_ANCRE, 10)
                W = E.Monde(399, "fixe")
                TX, Ty, TI = E.arrivees_idx(W, E.TAMPON, 0)
                n = 48 // k
                t0 = time.time()
                for _ in range(n):
                    idx = rng.integers(0, E.TAMPON, (k, E.NB))
                    pop.pas_seul(TX[idx], Ty[idx], 0.01, E.WD0)
                dur[f"K{k}"] = dict(s_par_pas=(time.time() - t0) / n)
        # une course courte entière, deux générations, avec ses mesures finales (le coût fixe d'une course)
        for b in ("SEUL_S", "K4_S"):
            t0 = time.time()
            R.course(b, "cnn", 398, gens=2)
            dur[f"{b}_2_generations_avec_mesures"] = time.time() - t0
        bilan["durees"] = dur
        print(json.dumps(dur, indent=1), flush=True)
    nom = "essai48_bilan_vrai.json" if VRAI else "essai48_bilan.json"
    json.dump(bilan, open(os.path.join(ICI, nom), "w"), indent=1)
    ok = all(all(x["identique"] and x["coupee"] for x in v) for k, v in bilan.items() if k.startswith("reprise_"))
    ok = ok and bon_identites(bilan)
    print("BILAN :", "bon" if ok else "ÉCART", flush=True)


if __name__ == "__main__":
    main()
