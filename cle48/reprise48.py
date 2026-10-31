"""Clé 48 — les courses du plan, avec des points de reprise.

Les onze bras de la clé 48 ont leurs points de reprise dans le moteur (echelle48.py, section « clé 48 ») :
  SEUL10   le réseau seul pesé à la 10e génération, nourri de tout son calcul d'entraînement (k = 1, div = 1) ;
  SEUL_H   le même, nourri de la moitié de ce calcul (k = 1, div = 2) ;
  PHOTO2   la recette coupée en deux réseaux qui se partagent la moitié du calcul (le maître, K2_H : k = 2, div = 2),
           puis la distillation de leur ensemble dans un seul réseau nourri de l'autre moitié (l'élève : eleve = 2) ;
  TOILE4   un seul réseau au taux cyclique, quatre cycles (la Toile de la clé 47) ;
  TOILE8   le même, huit cycles ;
  SEUL_Q, K2_Q, K4_Q   un, deux ou quatre réseaux qui se partagent le quart du calcul (div = 4) ;
  SEUL_S, K2_S, K4_S   un, deux ou quatre réseaux qui se partagent le seizième du calcul (div = 16).
"""
import echelle48 as E

BRAS48 = {
    "SEUL10": dict(k=1, div=1), "SEUL_H": dict(k=1, div=2), "PHOTO2": dict(k=2, div=2, eleve=2),
    "SEUL_Q": dict(k=1, div=4), "K2_Q": dict(k=2, div=4), "K4_Q": dict(k=4, div=4),
    "SEUL_S": dict(k=1, div=16), "K2_S": dict(k=2, div=16), "K4_S": dict(k=4, div=16),
}
TOILES48 = {"TOILE4": 4, "TOILE8": 8}


def course(bras, arch, seed, gens=E.GENS, finales=True, res=32, point=None, arret=None):
    """Une course du plan de la clé 48, avec points de reprise (point : le fichier ; arret : pour les essais)."""
    E.resolution(res)
    assert arch == "cnn"
    if bras in TOILES48:
        return E.course_toile48(arch, seed, 10, TOILES48[bras], gens, finales, point=point, arret=arret)
    c = BRAS48[bras]
    return E.course_coupe48(arch, seed, c["k"], c["div"], 10, gens, finales, c.get("eleve"), point=point, arret=arret)
