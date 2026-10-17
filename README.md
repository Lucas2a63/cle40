# Clé 40 — L'échelle : le même monde, en pleine résolution

Projet Singularité. La clé 39 a mis le **système entier** — une population de réseaux classée par un juge
extérieur et frais visé sur la classe fatale, qui pèse juste et laisse apprendre avant de peser — face à la
**recette standard** — un réseau seul, bien pesé et réglé —, à calcul d'entraînement égal, sur CIFAR-10
réduit en 16×16. Cette clé refait la même comparaison sur les images d'origine, en **32×32** : quatre fois
plus de pixels par image, des réseaux quatre fois plus coûteux par image, le même nombre de pas. À graine
égale, les courses voient les mêmes images, dans le même ordre, que celles de la clé 39 — en quatre fois
plus fin : on peut donc lire, graine par graine, ce que l'échelle change.

La classe fatale est le **chat** (3 % du monde, ×30 pour le juge). Six bras : le système entier (SYS), la
même population sans sélection (INDEP), un réseau seul avec tout le calcul (SEUL, et SEUL10 avec la
bascule), un réseau 32 fois plus large (GRAND), un réseau seul avec le calcul d'un seul membre de la
population (COURT). Convolutif sur 12 graines, perceptron sur 24.

Les prédictions, la règle de décision et la règle de réglage de la recette standard ont été écrites
**avant toute course** : `PREENREGISTREMENT_40.md`. Son empreinte et celles du code, du plan et des données
sont dans `MANIFESTE40.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données sont dans le dépôt, en morceaux de moins de 25 Mo : voir l'étape 0
de `CLAUDE.md`.

```sh
sha256sum -c MANIFESTE40.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle40.py                    # l'environnement calcule comme le bac à sable
python3 lancer40.py --tranche P1         # une tranche du plan
```

Le plan (`plan40.json`, écrit par `plan40.py`) : le balayage de réglage (tranches B1 à B3, graine 99), les
populations (P1 à P8) et, une fois les taux fixés par `choix40.py` dans `balayage40_choix.json`, les réseaux
seuls (R1 à R8). Chaque tranche peut tourner dans sa propre session cloud : les consignes des sessions sont
dans `CLAUDE.md`.

## Fichiers

- `echelle40.py` — le monde, les réseaux, la population, l'ancre, le juge, les mesures et les six bras ;
  la résolution (16 ou 32) n'y est plus écrite en dur.
- `identite40.py` — vérifie qu'en 16×16 il refait au bit près les courses publiées de la clé 39.
- `gradients40.py` — contrôle de la rétropropagation en 32×32 (différences finies, double précision).
- `pilote40.py` — durées et mémoire, sur une graine hors plan.
- `plan40.py`, `plan40.json` — les courses et leur répartition ; `choix40.py` — la règle de réglage.
- `lancer40.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler40.py` — réunit les résultats poussés sur toutes les branches.
- `controle40.py`, `controle40_attendu.json` — vérifie que l'environnement refait au bit près quatre débuts
  de course.
- `reference39.py`, `reference39.json` — les mesures des courses publiées de la clé 39, pour les tests
  d'échelle.
- `verdict40.py` — applique la règle de décision, telle qu'écrite.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — les images d'origine, 32×32, dans l'ordre des fichiers
(lots 1 à 5, puis le test). Les images 16×16 des clés 36 à 39 en sont la moyenne arrondie de chaque bloc 2×2.
