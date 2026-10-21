# Clé 44 — Une sélection qui serve l'ensemble

(Ce dossier `cle44/` du dépôt de la clé 40 contient toute la clé 44 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. À la clé 43, à seuil égal — chacun réglant son seuil du chat sur l'ancre, sans voir le juge —,
l'**ensemble** des 32 réseaux de la recherche aléatoire (une population sans sélection ; la moyenne de leurs
probabilités) battait les trois recettes standard remises à l'épreuve, et la sélection du système, qui réduit la
population à une seule lignée, lui coûtait 1,65 point. Cette clé demande si une **sélection peut servir l'ensemble** :
classer les réseaux selon ce qu'ils apportent à l'ensemble sur l'ancre, et non selon leur note propre — en copiant
les poids (bras ENS) ou les seuls réglages (bras ENSR). Elle mesure aussi, pour la première fois, la **diversité**
des réseaux d'une population (l'ambiguïté : l'écart quadratique moyen des probabilités des réseaux à leur moyenne).

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe sur le perceptron (3 072 → 128 →
10), en 32×32, sur les graines de la clé 43 (100 à 147). La recherche aléatoire, le système et les 32 recettes réglées
sont refaits : ils doivent redonner au bit près les courses de la clé 43, avec la diversité en plus.

Les prédictions et la règle de décision ont été écrites **avant toute course** : `PREENREGISTREMENT_44.md`. Son
empreinte et celles du code, du plan, des réglages et des données sont dans `MANIFESTE44.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir l'étape 0
de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE44.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle44.py                    # l'environnement calcule comme le bac à sable
python3 lancer44.py --tranche E1         # une tranche du plan
```

Le plan (`plan44.json`, écrit par `plan44.py`) : 240 courses — ENS, ENSR, INDEP, SYS et M32 sur les graines 100 à
147 —, en 20 tranches de 12 courses (E1 à E20). Chaque tranche peut tourner dans sa propre session cloud : les
consignes sont dans `CLAUDE.md`.

## Fichiers

- `echelle44.py` — le moteur de la clé 43, plus la diversité de la population sur le juge, le critère de sélection
  « apport à l'ensemble » et les bras ENS et ENSR.
- `identite44.py` — vérifie qu'il refait au bit près des courses publiées de la clé 43.
- `essai44.py` — vérifie que les bras et les mesures nouvelles tournent.
- `plan44.py`, `plan44.json` — les courses et leur répartition.
- `balayage40_choix.json`, `balayage41_choix.json` — les réglages des recettes standard (clés 40 et 41).
- `lancer44.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler44.py` — réunit les résultats poussés sur toutes les branches.
- `controle44.py`, `controle44_attendu.json` — vérifie que l'environnement refait au bit près quatre débuts de course.
- `verdict44.py` — applique la règle de décision, telle qu'écrite (il lit aussi les courses publiées de la clé 43,
  dont il vérifie les empreintes).
- `ana44.py` — l'analyse descriptive, écrite avant les courses.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1 à 5,
puis le test). Les mêmes fichiers que les clés 40 à 43.
