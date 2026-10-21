"""Le moteur de la clé 44 refait au bit près des courses publiées de la clé 43 (graine 100) :
  python3 identite44.py --indep   → identite44_indep.json : INDEP entière ;
  python3 identite44.py --sys     → identite44_sys.json : SYS entière ;
  python3 identite44.py --m32     → identite44_m32.json : M32 entière ;
  python3 identite44.py --court2  → identite44_court2.json : COURT2 entière, et les trois premières générations de
                                    SYS graine 103.
Les valeurs comparées sont toutes celles de la course publiée (hors durée et tâche) ; les mesures nouvelles de la clé 44
(la diversité) n'y figurent pas et sont comptées à part."""
import json, os, sys, glob
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle44 as E

REF = {}
for f in glob.glob(os.path.join(ICI, "..", "cle43", "resultats", "tranche_*.json")):
    REF.update(json.load(open(f)))
C40 = json.load(open(os.path.join(ICI, "balayage40_choix.json")))
C41 = json.load(open(os.path.join(ICI, "balayage41_choix.json")))


def plat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from plat(v, f"{p}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from plat(v, f"{p}[{i}]")
    else:
        yield p, x


def comparer(nouveau, ref, gens=None):
    if gens is not None:
        ref = dict(tr=ref["tr"][:gens], juge=[d for d in ref["juge"] if d["g"] < gens])
        nouveau = dict(tr=nouveau["tr"][:gens], juge=[d for d in nouveau["juge"] if d["g"] < gens])
    else:
        ref = {k: v for k, v in ref.items() if k not in ("duree", "tache")}
        nouveau = {k: v for k, v in nouveau.items() if k not in ("duree", "tache")}
    pa, pb = dict(plat(ref)), dict(plat(nouveau))
    manquent = sorted(set(pa) - set(pb))
    en_plus = sorted(set(pb) - set(pa))
    n = e = 0; ex = []
    for k in pa:
        if k not in pb:
            continue
        n += 1
        if pa[k] != pb[k] or type(pa[k]) != type(pb[k]):
            e += 1
            if len(ex) < 5: ex.append((k, pa[k], pb[k]))
    import re
    cles = sorted({re.sub(r"\[\d+\]", "[]", k) for k in en_plus})
    return n, e, ex, len(manquent), len(en_plus), cles[:12]


def reglages(bras):
    if bras in ("SEUL", "GRAND", "COURT"):
        return E.LR0 * float(C40[f"{bras}_mlp"]), E.WD0
    if bras in ("SEUL2", "GRAND2", "COURT2", "M32"):
        return E.LR0 * float(C41[bras]["mult"]), (E.WD0 if C41[bras]["wd"] == "WD0" else float(C41[bras]["wd"]))
    return E.LR0, E.WD0


if "--indep" in sys.argv:
    cas, nom_sortie = [("INDEP", 100, None)], "identite44_indep.json"
elif "--sys" in sys.argv:
    cas, nom_sortie = [("SYS", 100, None)], "identite44_sys.json"
elif "--m32" in sys.argv:
    cas, nom_sortie = [("M32", 100, None)], "identite44_m32.json"
else:
    cas, nom_sortie = [("COURT2", 100, None), ("SYS", 103, 3)], "identite44_court2.json"
out = {}
for bras, s, g in cas:
    lr0, wd = reglages(bras)
    r = E.course(bras, "mlp", s, gens=g or E.GENS, finales=g is None, lr0=lr0, res=32, wd=wd)
    r = json.loads(json.dumps(r))
    n, e, ex, man, plus, cles = comparer(r, REF[f"mlp_{bras}_s{s}"], g)
    out[f"mlp_{bras}_s{s}"] = dict(gens=g or E.GENS, valeurs=n, ecarts=e, exemples=ex, manquantes=man,
                                   nouvelles=plus, cles_nouvelles=cles)
    print(f"mlp_{bras}_s{s}", g or "entière", n, "valeurs,", e, "écarts,", man, "manquantes ;", plus, "nouvelles", cles,
          ex[:2], flush=True)
json.dump(out, open(os.path.join(ICI, nom_sortie), "w"), indent=1)
