# Clé 42 — À seuil égal

(Ce dossier `cle42/` du dépôt de la clé 40 contient toute la clé 42 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. À la clé 41, sur le perceptron en 32×32, réglées aussi sur la décroissance des poids, les
**recettes standard** ont battu le **système entier** — une population de 32 réseaux classée par une ancre fraîche
visée sur la classe fatale, avec sélection —, et, après coup, l'essentiel des écarts tenait au seuil de décision de
la classe fatale. Cette clé refait la comparaison **à seuil égal** — chaque réseau livré règle son seuil de la même
façon, sur l'ancre, sans voir le juge — sur 48 graines neuves, et donne au système un génome qui peut atteindre les
réglages des recettes réglées (élargi, ou parti de la recette).

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe sur le perceptron (3 072 → 128 →
10), en 32×32.

Les prédictions et la règle de décision ont été écrites **avant toute course** : `PREENREGISTREMENT_42.md`. Son
empreinte et celles du code, du plan, des réglages et des données sont dans `MANIFESTE42.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir l'étape 0
de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE42.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle42.py                    # l'environnement calcule comme le bac à sable
python3 lancer42.py --tranche Y1         # une tranche du plan
```

Le plan (`plan42.json`, écrit par `plan42.py`) : les populations (tranches Y1 à Y16), le meilleur de 32 recettes
réglées (Z1 à Z3), les grands réseaux (L1 à L6), le réseau seul et les réseaux courts (C1 à C4) — 528 courses, sur
les graines 48 à 95. Chaque tranche peut tourner dans sa propre session cloud : les consignes sont dans `CLAUDE.md`.

## Fichiers

- `echelle42.py` — le moteur de la clé 41, plus le front de l'ancre du réseau livré, le génome réglable de la
  population, et les bras SYSL (génome élargi) et SYSR (génome parti de la recette réglée).
- `identite42.py` — vérifie qu'il refait au bit près des courses publiées de la clé 41.
- `essai42.py` — vérifie que les nouveaux bras tournent.
- `plan42.py`, `plan42.json` — les courses et leur répartition.
- `balayage40_choix.json`, `balayage41_choix.json` — les réglages des recettes standard (clés 40 et 41).
- `lancer42.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler42.py` — réunit les résultats poussés sur toutes les branches.
- `controle42.py`, `controle42_attendu.json` — vérifie que l'environnement refait au bit près quatre débuts de course.
- `verdict42.py` — applique la règle de décision, telle qu'écrite.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1 à 5,
puis le test). Les mêmes fichiers que les clés 40 et 41.
