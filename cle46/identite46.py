"""Clé 46 — le moteur refait-il, sur le réseau convolutif, les courses publiées de la clé 40 ?
  python3 identite46.py CAS [CAS…] → identite46_CAS.json ; CAS parmi :
    indep3   — la recherche aléatoire (INDEP) convolutive, graine 0 : ses trois premières générations (le suivi de chaque
               génération et les passages du juge des générations 0, 1 et 2) ;
    court    — le réseau court convolutif, graine 0, course entière ;
    seul10   — le réseau seul convolutif pesé à la 10e génération, graine 0, course entière ;
    grand    — le grand réseau convolutif, graine 0, course entière.
Les valeurs comparées sont toutes celles de la course publiée de la clé 40 (hors durée et tâche) dans les parties
comparées ; les valeurs que le moteur a ajoutées depuis (clés 41 à 45) sont comptées à part. Les taux des recettes sont
ceux de la clé 40 (balayage40_choix.json)."""
import json, os, sys, glob, re
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle46 as E

REF = {}
for f in glob.glob(os.path.join(ICI, "..", "cle40", "resultats", "tranche_*.json")):
    REF.update(json.load(open(f)))
CHOIX40 = json.load(open(os.path.join(ICI, "balayage40_choix.json")))


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
    return dict(valeurs=n, ecarts=e, exemples=ex, manquantes=len(manquent), exemples_manquantes=manquent[:5],
                nouvelles=len(en_plus), cles_nouvelles=cles)


def cas(nom):
    if nom == "indep3":
        r = json.loads(json.dumps(E.course("INDEP", "cnn", 0, gens=3, finales=False)))
        ref = REF["cnn_INDEP_s0"]
        ref = dict(tr=ref["tr"][:3], juge=[j for j in ref["juge"] if j["g"] <= 2])
        r = dict(tr=r["tr"][:3], juge=[j for j in r["juge"] if j["g"] <= 2])
        return comparer(r, ref)
    bras = {"court": "COURT", "seul10": "SEUL10", "grand": "GRAND"}[nom]
    lr0 = E.LR0 * float(CHOIX40[f"{bras}_cnn"])
    r = json.loads(json.dumps(E.course(bras, "cnn", 0, finales=True, lr0=lr0, res=32)))
    return comparer(r, REF[f"cnn_{bras}_s0"])


if __name__ == "__main__":
    for nom in sys.argv[1:]:
        out = dict(cas=nom, **cas(nom))
        json.dump(out, open(os.path.join(ICI, f"identite46_{nom}.json"), "w"), ensure_ascii=False, indent=1)
        print(json.dumps(out, ensure_ascii=False), flush=True)
