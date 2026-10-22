"""Essai de reprise46 : une course faite d'une traite par le moteur (echelle46.course) et la même course coupée puis
reprise par reprise46.course doivent donner le même résultat, au bit près (toutes les valeurs, toutes les clés, dans le
même ordre). Graines hors plan (201, 202), courses raccourcies.

  python3 reprise46_essai.py orig BRAS GENS GRAINE SORTIE.json
  python3 reprise46_essai.py rep  BRAS GENS GRAINE POINT SORTIE.json [ARRET] [COPIE_FIN1]
       ARRET : génération après laquelle la course s'arrête (simule un redémarrage) ; « - » : aucun arrêt
       COPIE_FIN1 : copie le point écrit après la première étape des mesures finales (pour reprendre de là)
  python3 reprise46_essai.py comparer A.json B.json
Chaque appel est un processus neuf, comme après un redémarrage."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, shutil, sys, time
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

LR0S = {"INDEPC": None, "SEUL10": 0.0625, "SEUL": 0.0625, "GRAND": 0.5, "COURT": 0.5}


def ecrire(chemin, r):
    with open(chemin, "w") as f:
        json.dump(r, f)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "comparer":
        x, y = json.load(open(a[1])), json.load(open(a[2]))
        sx, sy = json.dumps(x), json.dumps(y)

        def plat(o, p=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from plat(v, f"{p}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    yield from plat(v, f"{p}[{i}]")
            else:
                yield p, o
        px, py = list(plat(x)), list(plat(y))
        ecarts = sum(1 for (k1, v1), (k2, v2) in zip(px, py) if k1 != k2 or type(v1) is not type(v2) or v1 != v2)
        print(json.dumps(dict(identiques=sx == sy, valeurs=len(px), valeurs_b=len(py), ecarts=ecarts)))
        sys.exit(0 if sx == sy else 1)
    import echelle46 as E
    bras, gens, graine = a[1], int(a[2]), int(a[3])
    lr0 = E.LR0 * LR0S[bras] if LR0S[bras] else E.LR0
    t0 = time.time()
    if a[0] == "orig":
        r = E.course(bras, "cnn", graine, gens=gens, finales=True, lr0=lr0, res=32)
        ecrire(a[4], r)
    else:
        import reprise46
        point, sortie = a[4], a[5]
        arret = None if len(a) < 7 or a[6] == "-" else int(a[6])
        if len(a) > 7:
            copie, ancien = a[7], reprise46._ecrire

            def ecrire_et_copier(chemin, obj):
                ancien(chemin, obj)
                if len(obj.get("fin", {})) == 1 and not os.path.exists(copie):
                    shutil.copyfile(chemin, copie)
            reprise46._ecrire = ecrire_et_copier
        try:
            r = reprise46.course(bras, "cnn", graine, gens=gens, finales=True, lr0=lr0, res=32, point=point, arret=arret)
            ecrire(sortie, r)
        except reprise46.Arret as e:
            print(f"arrêt simulé après la génération {e.args[0]}")
    print(f"{a[0]} {bras} {gens} gén. : {time.time() - t0:.0f} s")
