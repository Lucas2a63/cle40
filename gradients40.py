"""Contrôle des gradients en 32×32 : la rétropropagation d'echelle40 contre des différences finies centrées,
en double précision (le code calcule en float32 ; pour ce contrôle seulement, F32 devient float64).
python3 gradients40.py → « BON » si l'écart relatif le plus grand est sous 1e-5."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echelle40 as E

E.resolution(32)
E.F32 = np.float64
rng = np.random.default_rng(0)
pire = 0.0
for arch, dims in (("mlp", {}), ("cnn", {}), ("mlp", E.LARGE["mlp"]), ("cnn", E.LARGE["cnn"])):
    with E.taille(2, **dims):
        net = E.ARCHS[arch](np.random.default_rng(1))
        for k in net.POIDS:
            setattr(net, k, getattr(net, k).astype(np.float64))
        X = rng.normal(0, 1, (2, 3, 32, 32, 3))                 # un lot propre à chacun des deux réseaux
        G = rng.normal(0, 1, (2, 3, 10))
        z, f, cache = net.avant(X)
        gr = net.arriere(G, cache)
        loc = 0.0
        for k in net.POIDS:
            W = getattr(net, k)
            for _ in range(6):
                i = tuple(rng.integers(0, s) for s in W.shape)
                old = W[i]; h = 1e-5
                W[i] = old + h; zp = net.avant(X)[0]
                W[i] = old - h; zm = net.avant(X)[0]
                W[i] = old
                num = ((zp - zm) * G).sum() / (2 * h)
                loc = max(loc, abs(num - gr[k][i]) / max(1e-8, abs(num) + abs(gr[k][i])))
        pire = max(pire, loc)
        print(f"{arch} {dims or ''} : traits {f.shape[-1]}, écart relatif max {loc:.1e}", flush=True)
print("BON" if pire < 1e-5 else "ÉCART", f"{pire:.1e}")
