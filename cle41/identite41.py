"""Le moteur de la clé 41 refait au bit près des courses publiées de la clé 40 (bras existants) :
python3 identite41.py → identite41.json. Courses : mlp COURT s5 et mlp SEUL s7 entières ; trois premières
générations de mlp SYS s0, mlp INDEP s1, cnn SYS s2."""
import json, os, sys, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle41 as E

REF = {}
for f in glob.glob(os.path.join(ICI, "..", "cle40", "resultats", "tranche_[PR]*.json")):
    REF.update(json.load(open(f)))
CHOIX = json.load(open(os.path.join(ICI, "..", "cle40", "balayage40_choix.json")))


def plat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from plat(v, f"{p}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from plat(v, f"{p}[{i}]")
    else:
        yield p, x


def comparer(a, b, gens=None):
    """Compare toutes les valeurs (hors durée et tâche) ; pour un rejeu partiel, les générations communes."""
    n = e = 0; ex = []
    if gens is not None:
        a = dict(tr=a["tr"][:gens], juge=[d for d in a["juge"] if d["g"] < gens])
        b = dict(tr=b["tr"][:gens], juge=[d for d in b["juge"] if d["g"] < gens])
    else:
        a = {k: v for k, v in a.items() if k not in ("duree", "tache")}
        b = {k: v for k, v in b.items() if k not in ("duree", "tache")}
    pa, pb = dict(plat(a)), dict(plat(b))
    assert set(pa) == set(pb), sorted(set(pa) ^ set(pb))[:5]
    for k in pa:
        n += 1
        if pa[k] != pb[k] or type(pa[k]) != type(pb[k]):
            e += 1
            if len(ex) < 5: ex.append((k, pa[k], pb[k]))
    return n, e, ex


cas = [("COURT", "mlp", 5, None), ("SYS", "mlp", 0, 3), ("INDEP", "mlp", 1, 3), ("SYS", "cnn", 2, 3)]
if len(sys.argv) > 1 and sys.argv[1] == "--long":
    cas = [("SEUL", "mlp", 7, None)]
out = {}
for bras, arch, s, g in cas:
    lr0 = E.LR0 * float(CHOIX[f"{bras}_{arch}"]) if bras in ("SEUL", "COURT", "GRAND", "SEUL10") else E.LR0
    r = E.course(bras, arch, s, gens=g or E.GENS, finales=g is None, lr0=lr0, res=32)
    r = json.loads(json.dumps(r))
    n, e, ex = comparer(r, REF[f"{arch}_{bras}_s{s}"], g)
    out[f"{arch}_{bras}_s{s}"] = dict(gens=g or E.GENS, valeurs=n, ecarts=e, exemples=ex)
    print(f"{arch}_{bras}_s{s}", g or "entière", n, "valeurs,", e, "écarts", ex[:2], flush=True)
nom = "identite41_long.json" if len(sys.argv) > 1 else "identite41.json"
json.dump(out, open(os.path.join(ICI, nom), "w"), indent=1)
