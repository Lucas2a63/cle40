# Clé 48 — La moyenne des poids sur un taux cyclique, et le découpage quand le calcul manque

(Ce dossier `cle48/` du dépôt de la clé 40 contient toute la clé 48 ; les données sont les morceaux de la clé 40,
à la racine du dépôt.)

Projet Singularité. À la clé 47, à calcul d'entraînement égal et à seuil égal, la recette du réseau seul coupée en
deux, quatre ou huit réseaux, lus en ensemble, a battu le réseau seul, alors qu'aucun de ces réseaux, pris seul, ne
faisait mieux que lui ; et, sans prédiction, la moyenne des poids des instantanés d'un entraînement au taux cyclique
(la Toile) a battu le réseau seul au même calcul et au même coût d'usage. Cette clé reprend cette moyenne en la
prédisant, sur des graines neuves (TOILE4, et TOILE8 à huit cycles) ; essaie la distillation à calcul total égal
(PHOTO2 : le maître, deux réseaux, et l'élève se partagent le calcul du réseau seul ; SEUL_H, l'élève sans maître) ; et
refait le découpage quand le calcul manque : un, deux ou quatre réseaux qui se partagent le quart (SEUL_Q, K2_Q, K4_Q)
ou le seizième (SEUL_S, K2_S, K4_S) du calcul du réseau seul.

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Tout se passe en 32×32 : la partie A sur les
graines 124 à 147 (neuves), la partie B sur les graines 100 à 123, celles des clés 46 et 47, dont les courses publiées
de la clé 47 servent de référence, graine par graine.

Les prédictions et la règle de décision ont été écrites **avant toute course** : `PREENREGISTREMENT_48.md`. Son
empreinte et celles du code, du plan et des données sont dans `MANIFESTE48.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont à la racine du dépôt, en morceaux de moins de 10 Mo : voir l'étape 0
de `CLAUDE.md` (dans ce dossier).

```sh
sha256sum -c MANIFESTE48.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle48.py                    # l'environnement calcule comme le bac à sable
python3 lancer48.py --tranche J1         # une tranche du plan (avec points de reprise)
```

Le plan (`plan48.json`, écrit par `plan48.py`) : 264 courses — onze bras —, en 24 tranches (J1 à J24) qui portent
chacune une graine de la partie A et une de la partie B. Chaque tranche peut tourner dans sa propre session cloud : les
consignes sont dans `CLAUDE.md`.

## Fichiers

- `echelle48.py` — le moteur de la clé 47, inchangé (seul son texte d'en-tête est réécrit), et une section nouvelle,
  « clé 48 », pour ses bras (avec leurs points de reprise).
- `reprise48.py` — les courses du plan.
- `essai48.py`, `essai48_bilan.json`, `essai48_bilan_vrai.json` — les essais faits avant le plan : chaque bras coupé
  puis repris, identique à la même course faite d'une traite ; les chemins communs identiques aux courses des clés 46 et
  47 (K2, K4P, la Toile, le réseau seul à 1, 1/2, 1/4, 1/16 du calcul), aussi sur 12 générations ; l'élève de PHOTO2 et
  SEUL_H partis des mêmes poids, avec les mêmes paquets ; l'empreinte du monde, la même pour tous les bras d'une
  graine ; les durées.
- `plan48.py`, `plan48.json` — les courses et leur répartition.
- `lancer48.py` — exécute une tranche, avec points de reprise ; reprend là où elle s'était arrêtée.
- `rassembler48.py` — réunit les résultats poussés sur toutes les branches, et compare les exemplaires d'une course
  calculée deux fois.
- `controle48.py`, `controle48_attendu.json` — vérifie que l'environnement refait au bit près quatre débuts de course.
- `verdict48.py` — applique la règle de décision, telle qu'écrite (il lit aussi les courses réunies de la clé 47, dont
  il vérifie les empreintes : `python3 verdict48.py --cle47 DOSSIER`, le dossier des 24 fichiers `tranche_J*.json` de
  la clé 47, publiés dans son archive).
- `ana48.py` — l'analyse descriptive, écrite avant les courses.
- `synthese48.py` — les données de synthèse sur lesquelles verdict48.py et ana48.py ont été essayés.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers (lots 1 à 5,
puis le test). Les mêmes fichiers que les clés 40 à 47.
