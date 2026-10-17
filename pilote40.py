"""Pilote de durée et de mémoire en 32×32, sur la graine 98 (hors plan, hors balayage) : quelques générations de
chaque bras, chacune dans son propre processus. python3 pilote40.py → pilote40.json"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, multiprocessing as mp, resource, sys, time
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

CAS = [("SYS", "cnn", 2), ("INDEP", "cnn", 1), ("SEUL", "cnn", 1), ("GRAND", "cnn", 1), ("COURT", "cnn", 3),
       ("SYS", "mlp", 3), ("SEUL", "mlp", 1), ("GRAND", "mlp", 1), ("COURT", "mlp", 3)]


def un(c):
    import echelle40 as E
    bras, arch, gens = c
    E.resolution(32)
    E.donnees()
    t0 = time.time()
    r = E.course(bras, arch, 98, gens=gens, finales=False, res=32)
    t1 = time.time()
    return dict(bras=bras, arch=arch, gens=gens, s_par_gen=(t1 - t0) / gens, juges=len(r["juge"]),
                rss_mo=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def finales():
    """Durée des mesures finales d'une population convolutive (32 réseaux) et de son réseau livré."""
    import echelle40 as E
    import numpy as np
    E.resolution(32)
    W = E.Monde(98)
    pop = E.Population(np.random.default_rng(1), "cnn", "BAS", 30.0, 10)
    t0 = time.time(); E.mesures_finales(pop, W); t1 = time.time()
    E.finales_reseau(pop, W, 0); t2 = time.time(); E.finales_ensemble(pop, W); t3 = time.time()
    return dict(fin_pop=t1 - t0, fin_livre=t2 - t1, fin_ens=t3 - t2,
                rss_mo=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


if __name__ == "__main__":
    out = []
    ctx = mp.get_context("spawn")
    for c in CAS:
        with ctx.Pool(1) as p:
            r = p.apply(un, (c,))
        print(r, flush=True); out.append(r)
    with ctx.Pool(1) as p:
        f = p.apply(finales)
    print(f, flush=True)
    json.dump(dict(cas=out, finales_cnn=f), open(os.path.join(ICI, "pilote40.json"), "w"), indent=1)
