# Clé 46 — L'ensemble sur l'architecture forte

(Ce dossier `cle46/` du dépôt de la clé 40 contient toute la clé 46 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. Depuis la clé 41, toutes les clés ont porté sur le perceptron : à seuil égal, l'**ensemble** des
32 réseaux d'une recherche aléatoire (la moyenne de leurs probabilités) y bat chacune des recettes standard, et aucune
sélection essayée ne l'a établi plus haut. Mais à la clé 40, au seuil par défaut, un seul petit **réseau convolutif**
finissait six à sept points au-dessus de cet ensemble de perceptrons, et l'ensemble convolutif n'avait pas d'avance sur
le meilleur réseau convolutif seul. Cette clé met les deux à seuil égal, sur le réseau convolutif de la clé 40 : la
recherche aléatoire convolutive, son ensemble et le choix final de la clé 45, contre le réseau seul (pesé à la 10e
génération, ou dès le départ), le grand réseau et le réseau court de la clé 40.

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe en 32×32, sur 24 graines (100 à
123) ; le perceptron sert de référence, graine par graine, à travers les courses publiées des clés 43 et 45.

Les prédictions et la règle de décision ont été écrites **avant toute course** : `PREENREGISTREMENT_46.md`. Son
empreinte et celles du code, du plan, des réglages et des données sont dans `MANIFESTE46.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir l'étape 0
de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE46.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle46.py                    # l'environnement calcule comme le bac à sable
python3 lancer46.py --tranche H1         # une tranche du plan
```

Le plan (`plan46.json`, écrit par `plan46.py`) : 120 courses — cinq bras convolutifs (INDEPC, SEUL10, SEUL, GRAND,
COURT) sur les graines 100 à 123 —, en 12 tranches de deux graines (H1 à H12). Chaque tranche peut tourner dans sa
propre session cloud : les consignes sont dans `CLAUDE.md`.

## Fichiers

- `echelle46.py` — le moteur de la clé 45, inchangé hors son en-tête (les bras convolutifs, le choix final et les
  mesures des clés 42 à 45 y étaient déjà).
- `identite46.py`, `identite46_*.json` — vérifie qu'il refait au bit près, sur le convolutif, des courses publiées de
  la clé 40.
- `plan46.py`, `plan46.json` — les courses et leur répartition.
- `balayage40_choix.json` — les taux de départ des recettes convolutives, choisis par le balayage de la clé 40.
- `lancer46.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler46.py` — réunit les résultats poussés sur toutes les branches.
- `controle46.py`, `controle46_attendu.json` — vérifie que l'environnement refait au bit près trois débuts de course.
- `verdict46.py` — applique la règle de décision, telle qu'écrite (il lit aussi les courses publiées des clés 43 et
  45, dont il vérifie les empreintes).
- `ana46.py` — l'analyse descriptive, écrite avant les courses.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1 à 5,
puis le test). Les mêmes fichiers que les clés 40 à 45.
