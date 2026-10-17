# Clé 42 — À seuil égal

Pré-enregistrement, écrit le 18 octobre 2026 vers 1 h 56 (heure de Paris), avant toute course du plan.
Projet Singularité (Lucas).

## La question

À la clé 41, sur le perceptron en 32×32, réglées aussi sur la décroissance des poids, les recettes standard ont battu
le système entier, de 0,7 à 1,7 point de vérité ×30. Et, après coup, le front de réglage — le seul seuil du chat
déplacé, choisi sur le juge — a montré que l'essentiel des écarts tenait au seuil du chat : au meilleur seuil, le
système ne battait plus, parmi les recettes standard, que le réseau seul. Deux questions suivent, et cette clé prend
les deux :

1. **À seuil égal**, quand chaque réseau livré règle son seuil du chat de la même façon, sur l'ancre, sans jamais voir
   le juge, que reste-t-il des écarts ? Le diagnostic de la clé 41 tient-il avec un seuil qu'un système peut
   réellement régler, et sur des graines neuves ?
2. **Avec un génome qui peut atteindre les réglages des recettes réglées** — élargi, ou parti de la recette —, la
   population apporte-t-elle quelque chose sur une recette bien réglée, à seuil égal ?

## Ce que les courses des clés 40 et 41 montrent déjà (lu avant ce texte, sur les courses publiées)

- Au meilleur seuil (choisi sur le juge, après coup ; graines 0 à 47) : système 0,548 ; réseau court réglé 0,552,
  meilleure de 32 recettes réglées 0,552, grand réseau réglé 0,551, réseau seul réglé 0,548 ; grand réseau de la
  clé 40 0,554, réseau court de la clé 40 0,551 ; recherche aléatoire 0,538 ; système sans curation 0,540. Écarts
  appariés au système : réseau court réglé +0,38 point [+0,08 ; +0,69], meilleure de 32 +0,33 [+0,04 ; +0,62], grand
  réseau réglé +0,27 [−0,03 ; +0,58], réseau seul réglé −0,01 [−0,31 ; +0,29] ; grand réseau de la clé 40 +0,53
  [+0,24 ; +0,83], réseau court de la clé 40 +0,31 [0,00 ; +0,62] ; recherche aléatoire −1,06, sans curation −0,83.
- Ces réseaux disent presque tous « chat » trop rarement pour la vérité ×30 : le biais du meilleur seuil est positif
  dans 537 courses sur 576, nul dans presque toutes les autres ; en moyenne +0,9 pour le système, +1,3 pour le grand
  réseau et le réseau court de la clé 40, +0,9 pour son réseau seul, +0,2 à +0,5 pour les recettes réglées.
- Le génome du système tire la décroissance de départ entre 1e-5 et 1e-3 et la borne à 1e-2 ; la sélection la fait
  monter à 3e-4 en médiane à la génération 119, quand les recettes réglées en prennent 3e-2. Son taux, tiré entre
  0,02 et 0,2, monte vers 0,13 puis se recuit vers 0,017 (en moyenne sur les graines) ; la recette réglée du réseau
  court part de 0,0079 et descend en cosinus jusqu'à zéro.
- L'ancre de chaque génération : 200 images de la réserve dont 40 chats, repondérées pour estimer sans biais la
  vérité ×30 — le chat y pèse 48 %, comme pour le juge.

## Le montage

Tout se passe sur le **perceptron** (3 072 → 128 → 10), en **32×32**, dans le monde des clés 40 et 41 : CIFAR-10, le
chat fatal (3 % du monde, ×30 pour le juge), 500 images par génération tirées avec remise dans les 35 000 images du
flux, un tampon de 5 000, l'ancre de 200 images de la réserve dont 40 chats (réutilisées une fois la réserve épuisée),
le juge sur les 10 000 images de test, 120 générations, monde fixe. Le moteur (`echelle42.py`) est celui de la clé 41,
plus le front de l'ancre du réseau livré, un génome réglable pour la population, et deux bras ; pour les bras de la
clé 41, il refait leurs courses au bit près.

**L'unité de calcul** est le pas : un pas de gradient d'un réseau sur un lot de 64 images. Les bras de population et
le réseau seul dépensent 384 000 pas ; le grand réseau 12 000 pas, chacun 32 fois plus coûteux ; le réseau court
12 000 pas.

**Les bras**

| Bras | Ce que c'est |
|---|---|
| **SYS** | Le système entier des clés 36 à 41 : 32 réseaux ; l'ancre les note ; à chaque génération, le quart le moins bien noté est remplacé par des copies du quart le mieux noté (poids et préférences de curation copiés, réglages copiés puis mutés) ; génome de départ : taux 0,02 à 0,2, décroissance 1e-5 à 1e-3 (log-uniformes), bornes 1e-3 à 1 et 1e-7 à 1e-2. Livre le mieux noté à la dernière génération. |
| **SYSL** | Le même, **génome élargi** : taux de départ 0,002 à 0,2, décroissance de départ 1e-5 à 1e-1 ; bornes 1e-4 à 1 et 1e-7 à 1. La population de départ couvre ainsi les réglages des recettes réglées. |
| **SYSR** | Le même, **génome parti de la recette réglée** du réseau court : taux de départ entre LR0/8/√10 et LR0/8·√10 (autour de 0,0079), décroissance entre 3e-2/√10 et 3e-2·√10 (autour de 3e-2) ; bornes de SYSL. Ses réseaux gardent, comme dans le système, un taux constant au fil d'une génération, sans cosinus. |
| **INDEP** | La population de SYS sans sélection : la recherche aléatoire ; livre le mieux noté. |
| **SANSCUR** | Le système, avec une force de curation nulle. |
| **M32** | 32 réseaux courts réglés (taux LR0/8 en cosinus, décroissance 3e-2), côte à côte, sans sélection ; livre le mieux noté par l'ancre. |
| **SEUL2, GRAND2, COURT2** | Les recettes réglées sur deux axes à la clé 41 (`balayage41_choix.json`) : réseau seul (LR0/128, 3e-2), grand réseau (LR0/4, 3e-2), réseau court (LR0/8, 3e-2). |
| **GRAND, COURT** | Le grand réseau et le réseau court de la clé 40, à leurs taux (`balayage40_choix.json`), décroissance 1e-4. |

Pas de nouveau balayage : les réglages sont ceux des clés 40 et 41. À graine égale, SYS, SYSL, SYSR et SANSCUR font
les mêmes tirages de sélection (appliqués ou non), et SYS, SYSL et SYSR les mêmes tirages de départ, portés dans
d'autres intervalles.

**Le seuil réglé sur l'ancre** (fixé ici, appliqué par `verdict42.py`). À la fin de chaque course, le *front de
l'ancre* du réseau livré : sa note par l'ancre — la même note pondérée que pour la sélection — sur les ancres réunies
des dix dernières générations (2 000 images, dont 400 chats ; certaines répétées, la réserve étant épuisée), pour
chaque biais b ajouté au logit du chat, sur la grille du front du juge (−30, puis de −6 à 14 par pas de 0,25). Le
biais retenu est la médiane basse des biais qui maximisent cette note (le k-ième de leur liste triée, k = (nombre − 1)
÷ 2, en partant de zéro). La vérité au seuil réglé, **V_cal**, est la vérité ×30 du juge à ce biais :
V = w·Rf(b) + (1 − w)·Ao(b), avec w = 0,9 / 1,87, le poids du chat dans la vérité ×30 (Rf : le rappel du chat ;
Ao : le rappel moyen des neuf autres classes). Le juge n'entre jamais dans le choix du seuil. Pour les bras de
population, l'ancre de la génération 119, qui a désigné le réseau livré, fait partie des dix : c'est ce que le
système a sous la main.

**Les graines** : 48 graines neuves, 48 à 95, pour tous les bras. 528 courses ; environ 170 heures de calcul d'un
cœur.

**Les mesures.** Le *réseau livré* est le mieux noté par l'ancre pour les bras de population (SYS, SYSL, SYSR,
INDEP, SANSCUR, M32), le réseau lui-même pour les autres.

- **V_cal** : la vérité ×30 du juge pour le réseau livré, génération 119, au seuil réglé sur l'ancre (ci-dessus).
- **V_fin** : la même, au seuil que le réseau se donne (sans biais).
- **G_cal** = V_cal − V_fin : ce que rapporte le réglage du seuil.
- **H** : la note de l'ancre pour le réseau livré, moins sa vérité selon le juge, en moyenne sur les 11 passages 40,
  48, …, 112, 119.
- **lwd119** : le logarithme de la décroissance de la population (moyenne géométrique de ses 32 réseaux), à la
  génération 119.
- Rapportés sans prédiction : **V_max**, la vérité au meilleur seuil du juge (le diagnostic de la clé 41) ; le biais
  retenu et le meilleur biais ; V_vie (la vérité au seuil du réseau, en moyenne sur les 16 passages 0, 8, …, 119) ;
  E_fin (l'ensemble des 32 réseaux, au seuil par défaut) ; le rappel du chat, la précision ordinaire, les fausses
  alertes ; le banc de stress, la sonde, la dimension effective ; les lignées ; les trajectoires des réglages ; les
  durées.

## Les prédictions

Mon raisonnement. Le diagnostic de la clé 41 devrait tenir avec un seuil réglé sur l'ancre, un peu brouillé par le
bruit d'une ancre de 2 000 images : les écarts du meilleur seuil devraient garder leur signe et leur ordre de
grandeur. Le génome élargi laisse à la sélection la possibilité de trouver une décroissance forte ; mais la sélection
est myope — elle garde, à chaque génération, le mieux noté du moment, et une décroissance forte paie tard — : je
n'attends qu'un petit gain, s'il y en a un. Partir de la recette réglée devrait aider davantage le système ; mais ses
réseaux gardent un taux constant au fil d'une génération, sans le cosinus qui mène celui de la recette jusqu'à zéro,
et, à la clé 41, à rappel du chat égal, les populations reconnaissaient moins bien les autres classes que les
recettes standard : je ne crois pas qu'il batte la recette réglée à seuil égal.

**Principales** (corrigées ensemble, voir la règle) — toutes sur V_cal, graines 48 à 95 :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | V_cal(SYSR) > V_cal(COURT2) — à seuil égal, le système parti de la recette réglée bat la recette réglée du réseau court | +0 à +0,8 | 25 % |
| **P2** | V_cal(SYSR) > V_cal(SYS) — à seuil égal, partir de la recette réglée améliore le système | +0,2 à +1,0 | 55 % |
| **P3** | V_cal(SYS) > V_cal(INDEP) — à seuil égal, la sélection bat la recherche aléatoire | +0,5 à +1,5 | 90 % |
| **P4** | V_cal(COURT2) > V_cal(SYS) — à seuil égal, la recette réglée du réseau court bat le système | +0,1 à +0,8 | 60 % |

**Secondaires** (sur V_cal sauf mention) :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | V_cal(SYSR) > V_cal(M32) | +0 à +0,8 | 25 % |
| **S2** | V_cal(SYSR) > V_cal(SEUL2) | +0 à +0,8 | 35 % |
| **S3** | V_cal(SYSR) > V_cal(GRAND2) | +0 à +0,8 | 30 % |
| **S4** | V_cal(SYSL) > V_cal(SYS) — élargir le génome améliore le système | +0,1 à +0,6 | 40 % |
| **S5** | V_cal(SYSL) > V_cal(COURT2) | +0 à +0,5 | 15 % |
| **S6** | V_cal(M32) > V_cal(SYS) — la meilleure de 32 recettes réglées bat le système | +0,1 à +0,7 | 55 % |
| **S7** | V_cal(GRAND) > V_cal(SYS) — le grand réseau de la clé 40 bat le système | +0,2 à +0,8 | 65 % |
| **S8** | V_cal(COURT) > V_cal(SYS) — le réseau court de la clé 40 bat le système | +0,1 à +0,6 | 45 % |
| **S9** | V_fin(SYS) > V_fin(COURT) — au seuil du réseau, le gain de la clé 40 se retrouve sur un troisième jeu de graines | +0,8 à +2,5 | 90 % |
| **S10** | G_cal(COURT) > G_cal(SYS) — régler le seuil sur l'ancre profite plus au réseau court de la clé 40 qu'au système | +1 à +3 | 90 % |
| **S11** | V_cal(SYS) > V_cal(SANSCUR) — à seuil égal, la curation compte | +0,3 à +1,3 | 80 % |
| **S12** | \|V_cal(M32) − V_cal(COURT2)\| < 0,5 point — choisir par l'ancre la meilleure de 32 recettes réglées ne change pas la vérité de plus d'un demi-point | — | 70 % |
| **S13** | H(SYSR) > H(COURT2) — le réseau livré par le système parti de la recette se surestime, la recette non | +2 à +5 | 90 % |
| **S14** | lwd119(SYSL) > lwd119(SYS) — le génome élargi fait monter plus haut la décroissance de la population | — | 75 % |
| **S15** | V_cal(SEUL2) > V_cal(SYS) | +0 à +0,6 | 35 % |
| **S16** | V_cal(COURT2) > V_cal(COURT) — à seuil égal, régler aussi la décroissance améliore le réseau court | +0 à +0,4 | 30 % |

**Deux lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « à seuil égal, le gain du système sur les recettes de la clé 40 disparaît »** : sur V_cal, le système ne
  bat ni le grand réseau ni le réseau court de la clé 40 (p ≥ 0,025 unilatéral pour chacun). Je prédis que oui (70 %).
- **L2 — « à seuil égal, la population apporte quelque chose sur une recette bien réglée »** : sur V_cal, SYSR bat
  COURT2, SEUL2, GRAND2 et M32, chacun à p < 0,025 unilatéral. Je prédis que non (80 %).

**Comparaisons rapportées sans prédiction** (donc sans verdict) : V_max et ce que coûte le réglage sur l'ancre par
rapport au meilleur seuil (V_max − V_cal) ; les biais retenus et les meilleurs biais ; V_fin bras par bras ; SYSL et
SYSR contre les autres bras ; les ensembles ; V_vie ; les trajectoires des réglages et les lignées ; le rappel du
chat, la précision ordinaire et les fausses alertes ; le banc de stress, la sonde, la dimension effective ; la durée
des courses.

**Ce que chaque issue voudra dire.**

- Si L2 tient : à seuil égal et sur cette architecture, la population apporterait de la capacité au-delà d'une
  recette bien réglée — le premier signe robuste de ce genre.
- Si P2 tient mais pas P1 : partir de la recette aide le système, sans qu'il batte la recette.
- Si P4 et S6 tiennent : à seuil égal, le système reste derrière la recette réglée — ses réseaux apprennent moins
  bien, ce n'est pas seulement son seuil.
- Si L1 et S9 tiennent : le gain de la clé 40 était un effet de seuil, et un seuil réglé sur l'ancre — qu'un système
  peut régler sans le juge — suffit à le défaire, sur des graines neuves.
- Si S10 tient : le réglage sur l'ancre profite plus aux recettes de la clé 40 qu'au système, dont la sélection fait
  déjà une partie de ce travail.
- Si S14 tient sans S4 : le génome élargi fait monter la décroissance sans gain à seuil égal.

## La règle de décision

Toutes les comparaisons sont appariées par graine, sur les 48 graines. Pour chaque test, l'écart par graine
d = A − B, le t de Student et la valeur p unilatérale exacte à n − 1 = 47 degrés de liberté.

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ; *indécidable*
  sinon.
- **Tests secondaires (S1–S11, S13–S16)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ;
  *indécidable* sinon. Les p ajustés par Holm sur les 20 tests sont rapportés pour information.
- **Équivalence (S12)** : deux tests unilatéraux, bornes ±0,005, chacun à 0,025. *Confirmée* si les deux rejettent ;
  *démentie* si l'écart dépasse significativement une borne ; *indécidable* sinon.
- Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
- Plus de deux graines manquantes pour un bras : les tests qui l'utilisent sont *incomplets*.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %.

Le script `verdict42.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle, et calcule les
deux lectures.

## Ce qui a tourné avant ce texte

- `identite42.py` : le moteur de la clé 42 refait au bit près des courses publiées de la clé 41 — le réseau court
  réglé (graine 0), le réseau court de la clé 40 (graine 30) et la meilleure de 32 (graine 0) entiers, et les trois
  premières générations de SYS (graine 24), INDEP (25), SANSCUR (3) et POIDS (7) : 4 381 valeurs, aucun écart
  (`identite42.json`, `identite42_m32.json`). Le front de l'ancre, nouveau, s'ajoute aux sorties sans rien changer.
- `essai42.py` : deux générations de SYSL et de SYSR sur la graine 96 (hors plan), pour vérifier qu'ils tournent ;
  seules les durées et la forme des sorties ont été affichées.
- `controle42.py --ecrire` : les valeurs attendues du contrôle d'environnement, sur la graine 97 (hors plan) ; elles
  ont été écrites dans un fichier sans être affichées.
- `verdict42.py` a été essayé sur des données de synthèse construites à partir des courses de la clé 41 (graines
  renommées, front de l'ancre simulé) — aucune donnée de la clé 42.
- Rien du plan.

## Où et comment

Les tranches (Y1 à Y16 : populations ; Z1 à Z3 : M32 ; L1 à L6 : grands réseaux ; C1 à C4 : réseau seul et réseaux
courts) tournent dans des sessions cloud que je lance depuis l'ordinateur de Lucas, à sa demande, sur ses crédits ;
chacune exécute une tranche telle qu'écrite (`cle42/CLAUDE.md`) et pousse ses résultats sur sa branche du dépôt public
de la clé 40 (https://github.com/Lucas2a63/cle40), où la clé 42 occupe le dossier `cle42/` ; le bac à sable ne fait
que lire ce dépôt. Rien n'agit hors de ce cadre : pas d'agent autonome, pas d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Une seule architecture (le perceptron), un seul monde (fixe), une seule base d'images, des réseaux en numpy, sans
  augmentation de données, en descente de gradient simple.
- Les réglages des recettes viennent du balayage de la clé 41, sur une seule graine ; SYSR est centré sur la recette
  du réseau court (le départ de son cosinus), sans cosinus ; les intervalles de SYSL et de SYSR sont des choix.
- Le seuil réglé sur l'ancre repose sur 2 000 images, dont des répétitions ; pour les bras de population, l'ancre qui
  a désigné le réseau livré en fait partie.
- Le calcul est égal à l'entraînement, pas à l'usage ni au réglage (le balayage de la clé 41 n'est pas compté).
- Les interactions entre ingrédients ne sont pas testées.
