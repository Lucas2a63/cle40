# Clé 45 — Choisir l'ensemble à la fin

(Ce dossier `cle45/` du dépôt de la clé 40 contient toute la clé 45 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. À la clé 44, une sélection qui classe à chaque génération les réseaux selon leur apport à
l'**ensemble** (la moyenne de leurs probabilités) faisait un ensemble moins bon, d'environ un point, que ne pas
sélectionner du tout : la meilleure population, lue en ensemble, restait la recherche aléatoire. Cette clé essaie une
sélection qui ne touche pas à l'apprentissage : une fois les 120 générations finies, **choisir, sur l'ancre, quels
réseaux de la recherche aléatoire entrent dans l'ensemble** — la « sélection d'ensemble », une technique classique
(Caruana et coll., 2004) —, selon sept règles fixées d'avance (toutes, la sélection gloutonne avec remise, l'élagage,
les 8 et les 16 meilleurs pris un à un, autant de meilleurs que la sélection gloutonne en retient, autant tirés au
hasard).

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe sur le perceptron (3 072 → 128 →
10), en 32×32, sur les graines des clés 43 et 44 (100 à 147). La recherche aléatoire est refaite : elle doit redonner
au bit près ses courses de la clé 43, avec, en plus, le choix final.

Les prédictions et la règle de décision ont été écrites **avant toute course** : `PREENREGISTREMENT_45.md`. Son
empreinte et celles du code, du plan, des réglages et des données sont dans `MANIFESTE45.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir l'étape 0
de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE45.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle45.py                    # l'environnement calcule comme le bac à sable
python3 lancer45.py --tranche F1         # une tranche du plan
```

Le plan (`plan45.json`, écrit par `plan45.py`) : 48 courses — la recherche aléatoire suivie du choix final (INDEPC)
sur les graines 100 à 147 —, en 12 tranches de 4 courses (F1 à F12). Chaque tranche peut tourner dans sa propre
session cloud : les consignes sont dans `CLAUDE.md`.

## Fichiers

- `echelle45.py` — le moteur de la clé 44, plus le choix final (`choix_final`) et le bras INDEPC.
- `identite45.py` — vérifie qu'il refait au bit près une course publiée de la clé 43, et que la règle TOUS redonne
  son ensemble.
- `plan45.py`, `plan45.json` — les courses et leur répartition.
- `balayage40_choix.json`, `balayage41_choix.json` — les réglages des recettes standard (clés 40 et 41), repris pour
  l'analyse.
- `lancer45.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler45.py` — réunit les résultats poussés sur toutes les branches.
- `controle45.py`, `controle45_attendu.json` — vérifie que l'environnement refait au bit près deux débuts de course.
- `verdict45.py` — applique la règle de décision, telle qu'écrite (il lit aussi les courses publiées de la clé 43,
  dont il vérifie les empreintes).
- `ana45.py` — l'analyse descriptive, écrite avant les courses.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1 à 5,
puis le test). Les mêmes fichiers que les clés 40 à 44.

La sélection d'ensemble : R. Caruana, A. Niculescu-Mizil, G. Crew, A. Ksikes, *Ensemble Selection from Libraries of
Models*, ICML 2004.
