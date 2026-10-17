"""Le moteur de la clé 42 refait au bit près des courses publiées de la clé 41 (bras existants) :
python3 identite42.py → identite42.json. Courses : COURT2 graine 0 et COURT graine 30 entières, M32 graine 0
entière (python3 identite42.py --m32) ; trois premières générations de SYS graine 24, INDEP graine 25, SANSCUR
graine 3 et POIDS graine 7. Les valeurs comparées sont toutes celles de la course publiée (hors durée et tâche) ;
le front de l'ancre, nouveau, n'y figure pas."""
import json, os, sys, glob
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle42 as E

REF = {}
for f in glob.glob(os.path.join(ICI, "..", "cle41", "resultats", "tranche_[ADWM]*.json")):
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
    """Toutes les valeurs de la course publiée (hors durée et tâche) ; pour un rejeu partiel, les générations
    communes. Les clés nouvelles (le front de l'ancre) sont comptées à part."""
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
    return n, e, ex, len(manquent), len(en_plus), sorted({k.split("[")[0] for k in en_plus})[:6]


def reglages(bras):
    if bras in ("SEUL", "GRAND", "COURT"):
        return E.LR0 * float(C40[f"{bras}_mlp"]), E.WD0
    if bras in ("SEUL2", "GRAND2", "COURT2", "M32"):
        return E.LR0 * float(C41[bras]["mult"]), (E.WD0 if C41[bras]["wd"] == "WD0" else float(C41[bras]["wd"]))
    return E.LR0, E.WD0


cas = [("COURT2", 0, None), ("COURT", 30, None), ("SYS", 24, 3), ("INDEP", 25, 3), ("SANSCUR", 3, 3), ("POIDS", 7, 3)]
nom_sortie = "identite42.json"
if "--m32" in sys.argv:
    cas, nom_sortie = [("M32", 0, None)], "identite42_m32.json"
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
