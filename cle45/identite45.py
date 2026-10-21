"""Le moteur de la clé 45 refait au bit près la course publiée de la recherche aléatoire de la clé 43 (graine 100), et
la règle TOUS du choix final redonne l'ensemble des 32 réseaux de cette course :
  python3 identite45.py → identite45.json
Les valeurs comparées sont toutes celles de la course publiée (hors durée et tâche) ; le choix final (fin_choix),
nouveau, n'y figure pas : il est compté à part, et sa règle TOUS est comparée à fin_ens (front, front de l'ancre,
diversité)."""
import json, os, sys, glob, re
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle45 as E

REF = {}
for f in glob.glob(os.path.join(ICI, "..", "cle43", "resultats", "tranche_*.json")):
    REF.update(json.load(open(f)))


def plat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from plat(v, f"{p}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from plat(v, f"{p}[{i}]")
    else:
        yield p, x


def comparer(nouveau, ref):
    ref = {k: v for k, v in ref.items() if k not in ("duree", "tache")}
    nouveau = {k: v for k, v in nouveau.items() if k not in ("duree", "tache")}
    pa, pb = dict(plat(ref)), dict(plat(nouveau))
    manquent = sorted(set(pa) - set(pb))
    en_plus = sorted(set(pb) - set(pa))
    n = e = 0
    ex = []
    for k in pa:
        if k not in pb:
            continue
        n += 1
        if pa[k] != pb[k] or type(pa[k]) != type(pb[k]):
            e += 1
            if len(ex) < 5:
                ex.append((k, pa[k], pb[k]))
    cles = sorted({re.sub(r"\[\d+\]", "[]", k).split(".")[1] for k in en_plus})
    return n, e, ex, len(manquent), len(en_plus), cles


if __name__ == "__main__":
    r = E.course("INDEPC", "mlp", 100, gens=E.GENS, finales=True, lr0=E.LR0, res=32, wd=E.WD0)
    r = json.loads(json.dumps(r))
    n, e, ex, man, plus, cles = comparer(r, REF["mlp_INDEP_s100"])
    T, F = r["fin_choix"]["regles"]["TOUS"], r["fin_ens"]
    tous = dict(front=T["front"] == F["front"], ancre=T["ancre"]["note"] == F["ancre"]["note"],
                diversite=T["diversite"] == F["diversite"])
    out = dict(course="mlp_INDEPC_s100 contre mlp_INDEP_s100 (clé 43)", valeurs=n, ecarts=e, exemples=ex,
               manquantes=man, nouvelles=plus, cles_nouvelles=cles, TOUS_egal_fin_ens=tous)
    json.dump(out, open(os.path.join(ICI, "identite45.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))
