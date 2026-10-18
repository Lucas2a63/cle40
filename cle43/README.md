# Clé 43 — L'ensemble, à seuil égal

(Ce dossier `cle43/` du dépôt de la clé 40 contient toute la clé 43 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. À la clé 42, à seuil égal — chaque réseau livré réglant son seuil du chat sur l'ancre, sans voir
le juge —, le système entier n'avait plus d'avance établie sur aucune des six recettes standard essayées ; et, après
coup, la seule forme de population qui faisait nettement mieux qu'un réseau seul était l'**ensemble** des 32 réseaux
indépendants de la recherche aléatoire (la moyenne de leurs probabilités), au seuil par défaut, pour le coût d'usage du
grand réseau. Cette clé met l'ensemble à l'épreuve **à seuil égal** et **sous nos règles** — son seuil réglé sur
l'ancre, sa surestimation, la curation, la sélection —, sur 48 graines neuves, et remet à l'épreuve, à seuil égal, le
réseau seul de la clé 40.

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe sur le perceptron (3 072 → 128 →
10), en 32×32.

Les prédictions et la règle de décision ont été écrites **avant toute course** : `PREENREGISTREMENT_43.md`. Son
empreinte et celles du code, du plan, des réglages et des données sont dans `MANIFESTE43.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir l'étape 0
de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE43.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle43.py                    # l'environnement calcule comme le bac à sable
python3 lancer43.py --tranche E1         # une tranche du plan
```

Le plan (`plan43.json`, écrit par `plan43.py`) : les populations et leurs ensembles (tranches E1 à E16), le grand
réseau (L1 à L6), le réseau seul et le réseau court (C1 à C4) — 336 courses, sur les graines 100 à 147. Chaque tranche
peut tourner dans sa propre session cloud : les consignes sont dans `CLAUDE.md`.

## Fichiers

- `echelle43.py` — le moteur de la clé 42, plus le front du juge et le front de l'ancre de l'ensemble, la note de
  l'ancre de l'ensemble à chaque passage du juge, et le bras INDEP0 (la recherche aléatoire sans curation).
- `identite43.py` — vérifie qu'il refait au bit près des courses publiées des clés 41 et 42.
- `essai43.py` — vérifie que le bras et les mesures nouvelles tournent.
- `plan43.py`, `plan43.json` — les courses et leur répartition.
- `balayage40_choix.json`, `balayage41_choix.json` — les réglages des recettes standard (clés 40 et 41).
- `lancer43.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler43.py` — réunit les résultats poussés sur toutes les branches.
- `controle43.py`, `controle43_attendu.json` — vérifie que l'environnement refait au bit près quatre débuts de course.
- `verdict43.py` — applique la règle de décision, telle qu'écrite.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1 à 5,
puis le test). Les mêmes fichiers que les clés 40 à 42.
