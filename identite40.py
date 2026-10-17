"""Le moteur de la clé 40, en 16×16, refait-il au bit près les courses de la clé 39 ?
python3 identite40.py [cas ...]   (il faut donnees/cifar16_*.npz et ../cle39/resultats/)

Compare, valeur par valeur, des courses de la clé 39 (entières, ou leurs premières générations) à ce que
calcule echelle40 en 16×16. Seule différence permise : le champ calcul.res, nouveau."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import glob, json, sys, time
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle40 as E

C39 = os.path.join(ICI, "..", "cle39")
CHOIX = json.load(open(os.path.join(C39, "balayage39_choix.json")))
CAS = {  # nom : (bras, arch, graine, générations ou None pour la course entière)
    "mlp_COURT_s0": ("COURT", "mlp", 0, None),
    "cnn_COURT_s0": ("COURT", "cnn", 0, None),
    "mlp_SYS_s0": ("SYS", "mlp", 0, 3),
    "cnn_SYS_s0": ("SYS", "cnn", 0, 3),
    "mlp_SEUL_s0": ("SEUL", "mlp", 0, None),
    "cnn_INDEP_s1": ("INDEP", "cnn", 1, 3),
    "mlp_GRAND_s3": ("GRAND", "mlp", 3, None),
}


def feuilles(x, pre=""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from feuilles(x[k], f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from feuilles(v, f"{pre}[{i}]")
    else:
        yield pre, x


def stocke(nom):
    for f in sorted(glob.glob(os.path.join(C39, "resultats", "tranche_*.json"))):
        d = json.load(open(f))
        if nom in d:
            return d[nom]
    raise KeyError(nom)


def comparer(nom):
    bras, arch, s, gens = CAS[nom]
    ref = stocke(nom)
    lr0 = E.LR0 * (1.0 if bras in ("SYS", "INDEP") else float(CHOIX[f"{bras}_{arch}"]))
    t0 = time.time()
    r = E.course(bras, arch, s, gens=gens or E.GENS, finales=gens is None, lr0=lr0, res=16)
    r = json.loads(json.dumps(r))
    if gens is None:
        a = {k: v for k, v in r.items()}; b = {k: v for k, v in ref.items() if k not in ("duree", "tache")}
        a["calcul"] = {k: v for k, v in a["calcul"].items() if k != "res"}
    else:
        a = dict(tr=r["tr"], juge=r["juge"])
        b = dict(tr=ref["tr"][:gens], juge=[j for j in ref["juge"] if j["g"] < gens])
    fa, fb = dict(feuilles(a)), dict(feuilles(b))
    diff = [k for k in set(fa) | set(fb) if fa.get(k, "ABSENT") != fb.get(k, "ABSENT")]
    print(f"{nom:<14} {len(fb):>6} valeurs, {len(diff)} différentes ({time.time() - t0:5.0f} s)"
          + (f" — ex. {sorted(diff)[:3]}" if diff else ""), flush=True)
    return len(diff) == 0, len(fb)


if __name__ == "__main__":
    noms = sys.argv[1:] or list(CAS)
    res = {n: comparer(n) for n in noms}
    json.dump({n: dict(identique=ok, valeurs=v) for n, (ok, v) in res.items()},
              open(os.path.join(ICI, "identite40.json"), "w"), indent=1)
    print("IDENTIQUE" if all(ok for ok, _ in res.values()) else "DIFFÉRENCES", flush=True)
