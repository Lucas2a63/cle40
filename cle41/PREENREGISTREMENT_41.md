# Clé 41 — D'où vient le gain du perceptron

Pré-enregistrement, écrit le 17 octobre 2026 à 13 h 43 (heure de Paris), avant toute course du plan et avant le
balayage de réglage.
Projet Singularité (Lucas).

## La question

À la clé 40, sur le perceptron en 32×32, à calcul d'entraînement égal, le système entier — 32 réseaux, l'ancre
visée sur le fatal, la sélection — a battu chacune des recettes standard : le réseau seul qui reçoit tout le
calcul (+1,22 point), le grand réseau (+1,37), le réseau court (+1,46). C'était la barre écrite d'avance pour
parler d'un gain de capacité de la population. Deux questions suivent, et cette clé prend les deux :

1. **Le gain tient-il ?** Sur 24 graines neuves ; contre une recette standard réglée sur deux axes — le taux de
   départ et la décroissance des poids, quand la clé 40 ne réglait que le taux ; et contre la meilleure de 32
   recettes standard entraînées côte à côte et choisie par la même ancre que le système.
2. **D'où vient-il ?** En retirant au système, un à la fois, la copie des poids, l'évolution des réglages, et la
   curation.

## Ce que les courses de la clé 40 montrent déjà (lu avant ce texte, sur les courses publiées)

Sur le perceptron, la population du système descend d'un seul ancêtre dès la génération 40 environ (6 lignées
à la génération 8, 1,3 à la 40, 1 à la 119) : le réseau livré est le dernier d'une lignée dont les poids ont été
choisis à chaque génération parmi 32 variantes. Le taux d'apprentissage moyen monte d'abord (0,06 → 0,13 à la
génération 8), puis descend jusqu'à 0,016 ; la décroissance des poids monte de 1,1e-4 à 1,5e-3 — quinze fois
celle de la recette standard (1e-4, fixe, non réglée à la clé 40) ; la force de curation reste vers 0,5 à 0,7.
D'où trois explications possibles du gain, non exclusives : la copie des poids (une recherche guidée par
l'ancre), l'évolution des réglages (un taux qui se recuit, une décroissance qui monte), et une recette standard
mal réglée sur la décroissance.

## Le montage

Tout se passe sur le **perceptron** (3 072 → 128 → 10), en **32×32**, dans le monde de la clé 40 : CIFAR-10, le
chat fatal (3 % du monde, ×30 pour le juge), 500 images par génération tirées avec remise dans les 35 000 images
du flux, un tampon de 5 000, l'ancre de 200 images de la réserve dont 40 chats, le juge sur les 10 000 images de
test, 120 générations, monde fixe. Le moteur (`echelle41.py`) est celui de la clé 40, avec les nouveaux bras ;
pour les bras de la clé 40, il refait leurs courses au bit près.

**L'unité de calcul** est le pas : un pas de gradient d'un réseau sur un lot de 64 images. Les bras de
population (SYS, INDEP, POIDS, REGLAGES, SANSCUR, M32) et le réseau seul (SEUL, SEUL2) dépensent 384 000 pas ;
le grand réseau (GRAND, GRAND2) 12 000 pas, chacun 32 fois plus coûteux ; le réseau court (COURT, COURT2)
12 000 pas — le calcul d'un seul membre de la population.

**Les bras**

| Bras | Ce que c'est |
|---|---|
| **SYS** | Le système entier de la clé 40 : 32 réseaux ; l'ancre les note ; à chaque génération, le quart le moins bien noté est remplacé par des copies du quart le mieux noté — poids et préférences de curation copiés, réglages (taux, décroissance, force de curation) copiés puis mutés ; ×30 dans la perte dès le départ. Livre le mieux noté à la dernière génération. |
| **INDEP** | La même population sans sélection : une recherche aléatoire de réglages ; livre le mieux noté. |
| **POIDS** | La sélection copie l'état appris (poids et préférences de curation) ; chaque réseau garde ses réglages de départ, sans mutation. |
| **REGLAGES** | La sélection copie et mute les réglages ; chaque réseau garde ses poids et ses préférences. |
| **SANSCUR** | Le système entier, avec une force de curation nulle : chacun tire uniformément dans le tampon. |
| **M32** | 32 recettes standard côte à côte — taux en cosinus, décroissance fixe, tirage uniforme, ×30 dès le départ —, réglées comme COURT2, sans sélection ; livre la mieux notée par l'ancre à la dernière génération. Son ensemble (moyenne des 32) est aussi lu. |
| **SEUL, GRAND, COURT** | Les recettes standard de la clé 40, à leurs taux de la clé 40 (`balayage40_choix.json`), décroissance 1e-4. |
| **SEUL2, GRAND2, COURT2** | Les mêmes recettes, taux et décroissance réglés ensemble par un balayage. |

Les tirages de la sélection (choix des parents, mutations) sont faits dans tous les bras de population, puis
appliqués ou non : à graine égale, les bras partagent leurs tirages tant que leurs trajectoires ne divergent pas.

**Le réglage sur deux axes** (`choix41.py`, `plan41.py`), fixé ici avant tout balayage. Pour SEUL2, GRAND2 et
COURT2, une course sur la graine 99 (hors plan) par couple (taux, décroissance) : 5 taux, ×1/4 à ×4 autour du taux
retenu à la clé 40, et 5 décroissances, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 — soit 75 courses. La note d'un couple est
la moyenne des notes de l'ancre aux dix dernières générations (2 000 images), jamais le juge. Le couple retenu
est celui de la meilleure note. S'il est au bord de la grille dans une dimension, on essaie deux valeurs de plus
dans ce sens, l'autre réglage fixé à sa valeur retenue, au plus deux fois par dimension et par sens. M32 prend
les réglages de COURT2. Le script n'affiche que les réglages retenus ; je ne lirai les notes qu'après le verdict.

**Les graines** : 24 graines neuves, 24 à 47, pour la réplication (SYS, INDEP, SEUL, GRAND, COURT) ; les 48
graines 0 à 47 pour les nouveaux bras (POIDS, REGLAGES, SANSCUR, M32, SEUL2, GRAND2, COURT2). Pour les graines 0
à 23, les bras de la clé 40 sont ses courses publiées (`reference40.json`). 456 courses dans le plan, plus le
balayage (75 courses de départ) ; environ 150 heures de calcul d'un cœur.

**Les mesures** sont celles de la clé 40. Le *réseau livré* est le mieux noté par l'ancre pour les bras de
population, le réseau lui-même pour les autres.

- **V_fin** : la vérité ×30 du juge pour le réseau livré, à la génération 119.
- **V_vie** : la même, en moyenne sur les 16 passages du juge 0, 8, …, 112, 119.
- **H** : la note de l'ancre pour le réseau livré, moins sa vérité selon le juge, en moyenne sur les 11
  passages 40, 48, …, 112, 119.
- **E_fin** : la vérité ×30 de l'ensemble des 32 réseaux (moyenne des probabilités), génération 119.
- Rapportés sans prédiction : V_vie, le rappel du chat, la précision ordinaire, les fausses alertes, le banc de
  stress, la sonde, la dimension effective, les lignées, les trajectoires des réglages, les durées.

## Les prédictions

Mon raisonnement. Le gain de la clé 40 tient sur ses graines 12 à 23 (+1,4 à +1,6, lu après le verdict de la
clé 40) : je m'attends à ce qu'il se réplique. Sa source, je la crois d'abord dans la copie des poids : la
population ne garde qu'une lignée, et la fait avancer en gardant, à chaque génération, la meilleure de 32
variantes — une recherche que n'a ni un réseau seul, ni la meilleure de 32 recettes indépendantes. L'évolution
des réglages y ajoute un recuit et une décroissance plus forte ; mais une population qui copie les poids sans
muter ses réglages peut, à chaque génération, choisir parmi ses 32 réglages fixes celui qui convient : elle
devrait garder l'essentiel du gain. La curation, enfin, ne devrait pas peser lourd. Reste le doute le plus
sérieux : une recette standard réglée aussi sur la décroissance, ou la meilleure de 32 choisie par l'ancre,
pourrait rattraper une partie du système.

**Principales** (corrigées ensemble, voir la règle) :

| | Prédiction | Graines | Ampleur attendue | Confiance |
|---|---|---|---|---|
| **P1** | V_fin(SYS) > V_fin(COURT) — réplication : sur des graines neuves, le système bat le réseau court | 24–47 | +0,5 à +2,5 | 80 % |
| **P2** | V_fin(SYS) > V_fin(M32) — le système bat la meilleure de 32 recettes standard, choisie par l'ancre | 0–47 | +0,3 à +1,5 | 60 % |
| **P3** | V_fin(SYS) > V_fin(REGLAGES) — sans la copie des poids, le système perd | 0–47 | +0,5 à +2 | 70 % |
| **P4** | V_fin(POIDS) > V_fin(REGLAGES) — la copie des poids compte plus que l'évolution des réglages | 0–47 | +0,3 à +2 | 65 % |

**Secondaires** :

| | Prédiction | Graines | Ampleur attendue | Confiance |
|---|---|---|---|---|
| **S1** | V_fin(SYS) > V_fin(SEUL) — réplication | 24–47 | +0,3 à +2 | 75 % |
| **S2** | V_fin(SYS) > V_fin(GRAND) — réplication | 24–47 | +0,3 à +2,5 | 75 % |
| **S3** | V_fin(SYS) > V_fin(INDEP) — réplication : la sélection bat la recherche aléatoire | 24–47 | +0,5 à +3 | 80 % |
| **S4** | V_fin(SYS) > V_fin(SEUL2) | 0–47 | +0,2 à +1,5 | 55 % |
| **S5** | V_fin(SYS) > V_fin(GRAND2) | 0–47 | +0,2 à +2 | 60 % |
| **S6** | V_fin(SYS) > V_fin(COURT2) | 0–47 | +0,2 à +2 | 60 % |
| **S7** | V_fin(COURT2) > V_fin(COURT) — régler aussi la décroissance améliore le réseau court | 0–47 | +0,2 à +1,5 | 50 % |
| **S8** | V_fin(SEUL2) > V_fin(SEUL) — idem pour le réseau seul | 0–47 | +0,2 à +1,5 | 50 % |
| **S9** | V_fin(M32) > V_fin(COURT2) — choisir par l'ancre la meilleure de 32 recettes bat une recette | 0–47 | +0,1 à +0,8 | 55 % |
| **S10** | E_fin(M32) > V_fin(SYS) — l'ensemble des 32 recettes standard bat le réseau livré par le système | 0–47 | +2 à +5 | 85 % |
| **S11** | V_fin(REGLAGES) > V_fin(INDEP) — l'évolution des réglages, seule, bat la recherche aléatoire | 0–47 | +0,2 à +1,5 | 65 % |
| **S12** | V_fin(POIDS) > V_fin(INDEP) — la copie des poids, seule, bat la recherche aléatoire | 0–47 | +0,5 à +2,5 | 80 % |
| **S13** | \|V_fin(SYS) − V_fin(SANSCUR)\| < 1 point — la curation ne change pas la vérité finale de plus d'un point | 0–47 | — | 60 % |
| **S14** | H(SYS) > H(COURT2) — le réseau livré par le système se surestime, la recette standard non | 0–47 | +2 à +6 | 90 % |
| **S15** | H(M32) > H(COURT2) — choisir la meilleure de 32 par l'ancre la fait se surestimer | 0–47 | +1 à +4 | 80 % |
| **S16** | V_fin(SYS) > V_fin(POIDS) — l'évolution des réglages compte, en plus de la copie des poids | 0–47 | +0,2 à +1,5 | 45 % |

**Deux lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « la barre tient sur des graines neuves »** : sur les graines 24 à 47, le système bat SEUL, GRAND et
  COURT, chacun à p < 0,025 unilatéral. Je prédis que oui (65 %).
- **L2 — « la barre renforcée »** : sur les graines 0 à 47, le système bat SEUL2, GRAND2, COURT2 et M32, chacun à
  p < 0,025 unilatéral. Je prédis que non (55 %) : au moins une des recettes renforcées rattrapera le système.

**Comparaisons rapportées sans prédiction** (donc sans verdict) : les réglages retenus par le balayage ; les
bras de la décomposition entre eux et contre les recettes standard ; la part de l'écart entre le système et la
recherche aléatoire que retrouvent POIDS et REGLAGES ; les ensembles ; V_vie ; les trajectoires des réglages et
les lignées ; le rappel du chat, la précision ordinaire et les fausses alertes ; le banc de stress, la sonde, la
dimension effective ; la durée des courses.

**Ce que chaque issue voudra dire.**

- Si L1 tombe : le gain de la clé 40 tenait en partie aux graines 0 à 23 ; il ne faut pas s'appuyer dessus.
- Si L2 tient : le gain survit à une recette standard plus forte — réglée sur deux axes, ou choisie parmi 32 par
  la même ancre — ; ce serait, sur cette architecture, le premier signe robuste que la population apporte de la
  capacité à calcul égal.
- Si L2 tombe à cause de SEUL2, GRAND2 ou COURT2 : le gain venait, au moins en partie, d'une recette standard
  mal réglée sur la décroissance — le système aurait surtout servi de régleur automatique, ce qu'un praticien
  remplace par un balayage.
- Si L2 tombe à cause de M32 : le gain venait, au moins en partie, de ce qu'on choisit le meilleur de 32 réseaux
  avec une ancre — ce que fait aussi un praticien qui entraîne plusieurs fois et garde le meilleur sur une
  validation.
- P3, P4 et S16 diront quel ingrédient porte le gain — la copie des poids, ou l'évolution des réglages ; S13, la
  curation.

## La règle de décision

Toutes les comparaisons sont appariées par graine. Pour chaque test, l'écart par graine d = A − B, le t de
Student et la valeur p unilatérale exacte à n − 1 degrés de liberté (n = 24 sur les graines neuves, 48 sur
toutes).

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ;
  *indécidable* sinon.
- **Tests secondaires (S1–S12, S14–S16)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ;
  *indécidable* sinon. Les p ajustés par Holm sur les 20 tests sont rapportés pour information.
- **Équivalence (S13)** : deux tests unilatéraux, bornes ±0,01, chacun à 0,025. *Confirmée* si les deux
  rejettent ; *démentie* si l'écart dépasse significativement une borne ; *indécidable* sinon.
- Des écarts tous nuls (deux courses identiques) : t = 0, p = 0,5 dans les deux sens.
- Plus de deux graines manquantes pour un bras : les tests qui l'utilisent sont *incomplets*.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %.

Le script `verdict41.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle, et calcule
les deux lectures.

## Ce qui a tourné avant ce texte

- `identite41.py` : le moteur de la clé 41 refait au bit près des courses publiées de la clé 40 — le réseau
  court et le réseau seul du perceptron entiers, et les trois premières générations de deux populations du
  perceptron et d'une population convolutive : 2 388 valeurs, aucun écart (`identite41.json`, `identite41_long.json`).
- `essai41.py` : deux générations de chaque nouveau bras sur la graine 98 (hors plan, hors balayage), pour
  vérifier qu'ils tournent ; seules les durées et la forme des sorties ont été affichées.
- `controle41.py --ecrire` : les valeurs attendues du contrôle d'environnement (quatre débuts de course).
- `reference40.py` : les mesures des courses publiées de la clé 40 (graines 0 à 23).
- Rien du balayage, rien du plan.

## Où et comment

Le balayage (G1 à G7), la réplication (A1 à A6), la décomposition (D1 à D8) et, une fois les réglages fixés,
les recettes réglées sur deux axes (W1 à W6) et M32 (M1 à M3) tournent dans des sessions cloud que je lance
depuis l'ordinateur de Lucas, à sa demande, sur ses crédits ; chacune exécute une tranche telle qu'écrite
(`cle41/CLAUDE.md`) et pousse ses résultats sur sa branche du dépôt public de la clé 40
(https://github.com/Lucas2a63/cle40), où la clé 41 occupe le dossier `cle41/` — les données y sont déjà, en
morceaux, avec les empreintes de la clé 40 ; le bac à sable ne fait que lire ce dépôt, et fait tourner les
prolongements éventuels du balayage. Rien n'agit hors de ce cadre : pas d'agent autonome,
pas d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Une seule architecture (le perceptron), un seul monde (fixe), une seule base d'images, des réseaux en numpy,
  sans augmentation de données, en descente de gradient simple.
- Les bras de la décomposition retirent un ingrédient à la fois ; POIDS retire d'un coup le recuit du taux et la
  montée de la décroissance (ses réglages restent ceux du départ, tirés au hasard) ; les interactions ne sont
  pas testées.
- Le réglage est fait sur une seule graine ; M32 prend les réglages de COURT2, sans balayage propre.
- Le calcul est égal à l'entraînement, pas à l'usage : l'ensemble de M32 et le grand réseau coûtent 32 fois plus
  à interroger.
- Les graines 0 à 23 des bras de la clé 40 sont des courses déjà publiées ; les nouveaux bras y sont comparés à
  elles.
