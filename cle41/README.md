# Clé 41 — D'où vient le gain du perceptron

(Ce dossier `cle41/` du dépôt de la clé 40 contient toute la clé 41 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. À la clé 40, sur le perceptron en 32×32, le **système entier** — une population de 32
réseaux classée par un juge extérieur et frais visé sur la classe fatale, avec sélection — a battu, à calcul
d'entraînement égal, chacune des **recettes standard** (un réseau seul bien réglé, un réseau 32 fois plus grand,
un réseau seul avec le calcul d'un seul membre). Cette clé demande si ce gain tient — sur 24 graines neuves,
contre une recette standard réglée sur deux axes (taux et décroissance des poids), contre la meilleure de 32
recettes standard choisie par la même ancre — et d'où il vient : en retirant au système la copie des poids,
l'évolution des réglages, ou la curation.

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe sur le perceptron
(3 072 → 128 → 10), en 32×32 ; 48 graines pour les nouveaux bras, 24 graines neuves pour la réplication.

Les prédictions, la règle de décision et la règle de réglage ont été écrites **avant toute course** :
`PREENREGISTREMENT_41.md`. Son empreinte et celles du code, du plan et des données sont dans
`MANIFESTE41.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir
l'étape 0 de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE41.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle41.py                    # l'environnement calcule comme le bac à sable
python3 lancer41.py --tranche D1         # une tranche du plan
```

Le plan (`plan41.json`, écrit par `plan41.py`) : le balayage de réglage (tranches G1 à G7, graine 99), la
réplication (A1 à A6), la décomposition (D1 à D8) et, une fois les réglages fixés par `choix41.py` dans
`balayage41_choix.json`, les recettes réglées sur deux axes (W1 à W6) et la meilleure de 32 recettes (M1 à M3).
Chaque tranche peut tourner dans sa propre session cloud : les consignes des sessions sont dans `CLAUDE.md`.

## Fichiers

- `echelle41.py` — le moteur de la clé 40 et les nouveaux bras (POIDS, REGLAGES, SANSCUR, M32, SEUL2, GRAND2,
  COURT2).
- `identite41.py` — vérifie qu'il refait au bit près des courses publiées de la clé 40.
- `essai41.py` — vérifie que les nouveaux bras tournent.
- `plan41.py`, `plan41.json` — les courses et leur répartition ; `choix41.py` — la règle de réglage.
- `balayage40_choix.json` — les taux de la clé 40, pour SEUL, GRAND et COURT.
- `lancer41.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler41.py` — réunit les résultats poussés sur toutes les branches.
- `controle41.py`, `controle41_attendu.json` — vérifie que l'environnement refait au bit près quatre débuts de
  course.
- `reference40.py`, `reference40.json` — les mesures des courses publiées de la clé 40 (graines 0 à 23).
- `verdict41.py` — applique la règle de décision, telle qu'écrite.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1
à 5, puis le test). Les mêmes fichiers que la clé 40, dont c'est ici le dépôt.
