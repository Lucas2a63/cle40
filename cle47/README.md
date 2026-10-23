# Clé 47 — À calcul égal, un réseau ou plusieurs ?

(Ce dossier `cle47/` du dépôt de la clé 40 contient toute la clé 47 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. À la clé 46, sur le réseau convolutif et à seuil égal, le réseau seul nourri de tout le calcul
d'entraînement a battu l'ensemble des 32 réseaux de la recherche aléatoire de 1,9 point, sans que la clé sépare le
partage du calcul des réglages tirés au hasard. Cette clé coupe la meilleure recette convolutive en 2, 4, 8 ou 32
réseaux nourris chacun de 1/k du calcul, lus en ensemble (K2, K4, K8, K32) ; essaie une population aux réglages faits
pour le convolutif, sans curation (POPC) ; et quatre façons de faire entrer le multiple dans un seul réseau, ou de le
tirer d'un seul entraînement : la moyenne des poids le long de l'entraînement (VEILLEUR : SWA et EMA), la moyenne des
poids d'enfants d'un même parent (SOUPE), un ensemble distillé dans un seul réseau (PHOTO, avec K4 : K4P) et les
instantanés d'un entraînement au taux cyclique (TOILE).

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe en 32×32, sur les 24 graines de la
clé 46 (100 à 123), dont les courses publiées servent de référence, graine par graine.

Les prédictions et la règle de décision ont été écrites **avant toute course** : `PREENREGISTREMENT_47.md`. Son
empreinte et celles du code, du plan et des données sont dans `MANIFESTE47.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir l'étape 0
de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE47.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle47.py                    # l'environnement calcule comme le bac à sable
python3 lancer47.py --tranche J1         # une tranche du plan (avec points de reprise)
```

Le plan (`plan47.json`, écrit par `plan47.py`) : 192 courses — huit bras sur les graines 100 à 123 —, en 24 tranches
d'une graine (J1 à J24). Chaque tranche peut tourner dans sa propre session cloud : les consignes sont dans
`CLAUDE.md`.

## Fichiers

- `echelle47.py` — le moteur de la clé 46, inchangé, et une section nouvelle, « clé 47 », pour ses bras (avec leurs
  points de reprise).
- `reprise47.py` — les courses du plan (la population POPC passe par le corps de course_pop recopié, avec points de
  reprise, de la clé 46).
- `essai47.py`, `essai47_bilan.json`, `essai47_bilan_vrai.json` — les essais faits avant le plan : chaque bras coupé
  puis repris, identique à la même course faite d'une traite ; K1, le VEILLEUR et le parent de la SOUPE identiques à la
  course SEUL10 du moteur de la clé 46 ; les durées.
- `plan47.py`, `plan47.json` — les courses et leur répartition.
- `lancer47.py` — exécute une tranche, avec points de reprise ; reprend là où elle s'était arrêtée.
- `rassembler47.py` — réunit les résultats poussés sur toutes les branches, et compare les exemplaires d'une course
  calculée deux fois.
- `controle47.py`, `controle47_attendu.json` — vérifie que l'environnement refait au bit près trois débuts de course.
- `verdict47.py` — applique la règle de décision, telle qu'écrite (il lit aussi les courses publiées de la clé 46,
  dont il vérifie les empreintes).
- `ana47.py` — l'analyse descriptive, écrite avant les courses.
- `synthese47.py` — les données de synthèse sur lesquelles verdict47.py et ana47.py ont été essayés.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1 à 5,
puis le test). Les mêmes fichiers que les clés 40 à 46.
