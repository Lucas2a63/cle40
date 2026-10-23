"""Rassemble les résultats de la clé 47 poussés sur toutes les branches du dépôt.

  python3 rassembler47.py          toutes les tranches ; écrit resultats/tranche_K.json, resultats/bilan.txt et
                                   resultats/doublons.json
  python3 rassembler47.py K        la seule tranche K (reprise d'une tranche interrompue)

À lancer depuis le dossier cle47/, après : git fetch origin '+refs/heads/*:refs/remotes/origin/*'
Chaque session pousse sur sa propre branche. Une course trouvée sur plusieurs branches (ou déjà dans le fichier local)
est comparée, exemplaire par exemplaire, hors « duree » (le temps de calcul) : identiques, un seul est gardé ; s'ils
diffèrent, le premier trouvé est gardé et la différence est signalée (affichée, et écrite dans
resultats/doublons.json) — une course est déterministe à sa graine près.
"""
import json, os, subprocess, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
from plan47 import nom

plan = json.load(open(os.path.join(ICI, "plan47.json")))
os.makedirs(os.path.join(ICI, "resultats"), exist_ok=True)


def branches():
    r = subprocess.run(["git", "for-each-ref", "--format=%(refname)", "refs/remotes/"],
                       capture_output=True, text=True, cwd=ICI)
    return [b for b in r.stdout.split() if not b.endswith("/HEAD")]


def sans_duree(v):
    return json.dumps({k: x for k, x in v.items() if k != "duree"}, sort_keys=True)


def reunir(k, bs, doublons):
    chemin = os.path.join(ICI, "resultats", f"tranche_{k}.json")
    out = json.load(open(chemin)) if os.path.exists(chemin) else {}
    origine = {i: "local" for i in out}
    for b in bs:
        r = subprocess.run(["git", "show", f"{b}:./resultats/tranche_{k}.json"], capture_output=True, cwd=ICI)
        if r.returncode != 0:
            continue
        try:
            lus = json.loads(r.stdout)
        except ValueError:
            continue
        for i, v in lus.items():
            if i not in out:
                out[i], origine[i] = v, b
            elif sans_duree(out[i]) == sans_duree(v):
                doublons["identiques"] += 1
            else:
                doublons["differents"].append(dict(course=i, garde=origine[i], autre=b))
                print(f"DOUBLON DIFFÉRENT : {i} ({origine[i]} contre {b})", flush=True)
    if out:
        tmp = chemin + ".tmp"
        json.dump(out, open(tmp, "w")); os.replace(tmp, chemin)
    return out


def cle(k):
    return int(k[1:])


if __name__ == "__main__":
    bs = branches()
    ks = [sys.argv[1]] if len(sys.argv) > 1 else sorted(plan["tranches"], key=cle)
    doublons = dict(identiques=0, differents=[])
    lignes, total, faites = [], 0, 0
    for k in ks:
        taches = plan["tranches"][k]
        out = reunir(k, bs, doublons)
        n = sum(nom(t) in out for t in taches)
        total += len(taches); faites += n
        etat = "complète" if n == len(taches) else ("absente" if n == 0 else "incomplète")
        lignes.append(f"tranche {k:>3} : {n:>2}/{len(taches)}  {etat}")
    lignes.append(f"TOTAL : {faites}/{total} courses ({len(bs)} branches lues) ; doublons identiques : "
                  f"{doublons['identiques']}, différents : {len(doublons['differents'])}")
    if len(ks) > 1:
        open(os.path.join(ICI, "resultats", "bilan.txt"), "w").write("\n".join(lignes) + "\n")
        json.dump(doublons, open(os.path.join(ICI, "resultats", "doublons.json"), "w"), indent=1)
    print("\n".join(lignes))
