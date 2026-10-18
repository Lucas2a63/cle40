# Clé 43 — L'ensemble, à seuil égal

Pré-enregistrement, écrit le 18 octobre 2026 vers 13 h 48 (heure de Paris), avant toute course du plan.
Projet Singularité (Lucas).

## La question

À la clé 42, à seuil égal — chaque réseau livré réglant son seuil du chat sur l'ancre, sans voir le juge —, le système
entier n'avait plus d'avance établie sur aucune des six recettes standard essayées, et le grand réseau de la clé 40 le
battait. Après coup, la seule forme de population qui faisait nettement mieux qu'un réseau seul était l'**ensemble**
des 32 réseaux indépendants de la recherche aléatoire — la moyenne de leurs probabilités : 0,569 au seuil par défaut,
contre 0,551 pour le meilleur réseau seul à seuil réglé (le grand réseau de la clé 40), et 1,5 point de plus que ce
grand réseau même à son meilleur seuil du juge — pour le même coût d'usage (32 petits réseaux font autant de
multiplications par image que le grand) et le même calcul d'entraînement. Mais l'ensemble n'avait été lu qu'au seuil
par défaut, après coup, et sur des graines où on l'avait vu. Trois questions, que cette clé prend ensemble :

1. **À seuil égal** — l'ensemble réglant son seuil sur l'ancre comme les réseaux seuls —, et sur des graines neuves,
   l'ensemble de réseaux indépendants bat-il le grand réseau, au même coût d'usage, et les autres recettes ?
2. **Sous nos règles** : l'ensemble se surestime-t-il (sa note de l'ancre contre sa vérité) ? La curation l'aide-t-elle,
   la sélection le dessert-elle ? Fait-il mieux que l'ensemble de 32 recettes réglées identiques ?
3. **Le réseau seul de la clé 40**, que la clé 41 voyait encore battu par le système au meilleur seuil (+0,6 point) et
   que la clé 42 n'avait pas remis à l'épreuve, l'est-il à seuil égal ?

## Ce que les clés 41 et 42 montrent déjà (lu avant ce texte, sur les courses publiées)

- Clé 42 (graines 48 à 95), au seuil par défaut, les ensembles : recherche aléatoire 0,569 (rappel du chat 0,86,
  précision ordinaire 0,295) ; 32 recettes réglées 0,553 ; système 0,546 ; système sans curation 0,503. Sur toute la
  vie (16 passages du juge), au seuil par défaut : l'ensemble de la recherche aléatoire 0,545, le grand réseau 0,465.
- Clé 42, réseaux seuls à seuil réglé sur l'ancre : grand réseau de la clé 40 0,551, réseau court réglé 0,550, réseau
  livré par le système 0,547, par la recherche aléatoire 0,535 ; ces réseaux rattrapent alors de 91 à 93 % des chats,
  avec une précision ordinaire de 0,17 à 0,22.
- Clé 41 (graines 0 à 47), au meilleur seuil du juge : système 0,548, réseau seul de la clé 40 0,542 (+0,60 point pour
  le système, intervalle [+0,30 ; +0,89]) ; au seuil par défaut, 0,532 et 0,518.
- Les réseaux seuls, que personne ne choisit, ne se surestiment pas (note de l'ancre moins vérité : de −0,4 à −0,3
  point à la clé 42) ; le réseau livré par le système se surestime de 3,8 points, celui de la recherche aléatoire de
  4,7.

## Le montage

Tout se passe sur le **perceptron** (3 072 → 128 → 10), en **32×32**, dans le monde des clés 40 à 42 : CIFAR-10, le
chat fatal (3 % du monde, ×30 pour le juge), 500 images par génération tirées avec remise dans les 35 000 images du
flux, un tampon de 5 000, l'ancre de 200 images de la réserve dont 40 chats (réutilisées une fois la réserve épuisée),
le juge sur les 10 000 images de test, 120 générations, monde fixe. Le moteur (`echelle43.py`) est celui de la clé 42,
plus le front du juge et le front de l'ancre de l'ensemble, la note de l'ancre de l'ensemble à chaque passage du juge,
et un bras ; pour les bras des clés 41 et 42, il refait leurs courses au bit près.

**L'unité de calcul** est le pas : un pas de gradient d'un réseau sur un lot de 64 images. Les bras de population et
le réseau seul dépensent 384 000 pas ; le grand réseau 12 000 pas, chacun 32 fois plus coûteux ; le réseau court
12 000 pas. **Le coût d'usage** : un ensemble de 32 petits réseaux fait autant de multiplications par image que le grand
réseau (12 623 872) ; un petit réseau seul, 32 fois moins.

**Les bras**

| Bras | Ce que c'est |
|---|---|
| **SYS** | Le système entier des clés 36 à 42 : 32 réseaux ; l'ancre les note ; à chaque génération, le quart le moins bien noté est remplacé par des copies du quart le mieux noté (poids et préférences de curation copiés, réglages copiés puis mutés) ; génome des clés 36 à 41. |
| **INDEP** | La même population, sans sélection : la recherche aléatoire, avec curation (32 réseaux indépendants, réglages tirés au départ). |
| **INDEP0** | La recherche aléatoire **sans curation** : la même population, force de curation nulle (tirage uniforme dans le tampon). |
| **M32** | 32 réseaux courts réglés (taux LR0/8 en cosinus, décroissance 3e-2), côte à côte, sans sélection ni curation. |
| **GRAND** | Le grand réseau de la clé 40 (taux de la clé 40, décroissance 1e-4) : le même coût d'usage et le même calcul d'entraînement qu'une population. |
| **SEUL** | Le réseau seul de la clé 40 (taux de la clé 40, décroissance 1e-4), qui reçoit tout le calcul. |
| **COURT2** | Le réseau court réglé sur deux axes à la clé 41 (LR0/8, 3e-2). |

Pour les quatre bras de population, deux choses sont mesurées : le **réseau livré** (le mieux noté par l'ancre à la
dernière génération) et l'**ensemble** de ses 32 réseaux (la moyenne de leurs probabilités, puis la classe la plus
probable). Pas de nouveau balayage : les réglages sont ceux des clés 40 et 41. À graine égale, SYS, INDEP et INDEP0
font les mêmes tirages de départ (INDEP0 met ensuite la curation à zéro).

**Le seuil réglé sur l'ancre** (fixé ici, appliqué par `verdict43.py`), pour un réseau comme pour un ensemble. À la fin
de chaque course, le front de l'ancre : la note par l'ancre — la même note pondérée que pour la sélection — sur les
ancres réunies des dix dernières générations (2 000 images, dont 400 chats), pour chaque biais b de la grille du front
du juge (−30, puis de −6 à 14 par pas de 0,25). Pour un réseau, b s'ajoute au logit du chat ; pour un ensemble, au
logarithme de sa probabilité moyenne du chat (la classe retenue maximise log p̄ + b·[chat] — pour un réseau seul, c'est
exactement ajouter b à son logit). Le biais retenu est la médiane basse des biais qui maximisent cette note (le k-ième
de leur liste triée, k = (nombre − 1) ÷ 2, en partant de zéro). La vérité au seuil réglé est la vérité ×30 du juge à
ce biais : V = w·Rf(b) + (1 − w)·Ao(b), avec w = 0,9 / 1,87 (Rf : le rappel du chat ; Ao : le rappel moyen des neuf
autres classes). Le juge n'entre jamais dans le choix du seuil. Pour les populations, les ancres des dix dernières
générations ont aussi servi à la sélection (SYS) ou au choix du réseau livré.

**Les graines** : 48 graines neuves, 100 à 147, pour tous les bras (les graines 96 à 99 ont servi aux essais, aux
contrôles et aux balayages des clés 40 à 42). 336 courses ; environ 110 heures de calcul d'un cœur.

**Les mesures.**

Du réseau livré (pour les bras de population) ou du réseau lui-même (GRAND, SEUL, COURT2) :
- **V_cal** : la vérité ×30 du juge à la génération 119, au seuil réglé sur l'ancre (ci-dessus).
- **V_fin** : la même, au seuil que le réseau se donne (sans biais) ; **V_vie** : au seuil du réseau, en moyenne sur les
  16 passages 0, 8, …, 112, 119.
- **H** : la note de l'ancre moins la vérité du juge, en moyenne sur les 11 passages 40, 48, …, 112, 119.

De l'ensemble (bras de population) :
- **E_cal** : la vérité ×30 du juge pour l'ensemble, à la génération 119, au seuil réglé sur l'ancre.
- **E_fin** : la même, au seuil par défaut ; **EG_cal** = E_cal − E_fin ; **E_vie** : au seuil par défaut, en moyenne sur
  les 16 passages.
- **H_ens** : la note de l'ancre de l'ensemble moins sa vérité selon le juge, en moyenne sur les 11 passages 40, …, 119.

Rapportés sans prédiction : V_max et E_max (au meilleur seuil du juge) ; les biais retenus et les meilleurs biais ; le
front de l'ensemble à rappel du chat égal ; le banc de stress de l'ensemble ; V_fin, E_fin et H_ens bras par bras ; les
trajectoires des réglages et les lignées ; les durées.

## Les prédictions

Mon raisonnement. À la clé 42, au seuil par défaut, l'ensemble de la recherche aléatoire rattrapait déjà 86 % des chats
avec une précision ordinaire de 0,295, quand les réseaux seuls réglés sur l'ancre se tenaient vers 91 à 93 % des chats
et 0,17 à 0,22 : son front est vraisemblablement au-dessus du leur, et le réglage de son seuil sur l'ancre devrait le
pousser vers plus de chats sans le faire baisser. Je m'attends donc à ce qu'il batte, à seuil égal, le grand réseau au
même coût d'usage, d'un à deux points. La diversité des réglages — taux et décroissance tirés au hasard — devrait le
placer au-dessus de l'ensemble de 32 recettes identiques ; la sélection, qui réduit la population à une seule lignée,
devrait vider l'ensemble du système — lequel devrait pourtant faire un peu mieux que son propre réseau livré, et à peu
près aussi bien, à rappel du chat égal, que l'ensemble des 32 recettes réglées (S12 est à pile ou face). Je ne sais pas
ce que fait la curation à l'ensemble. Personne ne choisit
l'ensemble par l'ancre : il ne devrait pas se surestimer. Et la part du gain de la clé 40 qui survivait au meilleur
seuil face au réseau seul devrait survivre à seuil égal, le réglage sur l'ancre coûtant à peu près la même chose à
tous les bras à la clé 42.

**Principales** (corrigées ensemble, voir la règle) — graines 100 à 147 :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | E_cal(INDEP) > V_cal(GRAND) — à seuil égal, l'ensemble de la recherche aléatoire bat le grand réseau, au même coût d'usage | +1,0 à +2,5 | 85 % |
| **P2** | E_cal(INDEP) > E_cal(M32) — à seuil égal, l'ensemble de la recherche aléatoire bat l'ensemble de 32 recettes réglées | +0,5 à +1,8 | 75 % |
| **P3** | E_cal(INDEP) > V_cal(SYS) — à seuil égal, l'ensemble de la recherche aléatoire bat le réseau livré par le système | +1,2 à +2,5 | 90 % |
| **P4** | V_cal(SYS) > V_cal(SEUL) — à seuil égal, le système bat le réseau seul de la clé 40 | +0,2 à +0,9 | 70 % |

**Secondaires** :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | E_cal(INDEP) > V_cal(COURT2) — l'ensemble bat la recette réglée du réseau court | +0,8 à +2,0 | 85 % |
| **S2** | E_cal(INDEP) > E_cal(SYS) — l'ensemble d'une population sans sélection bat celui du système | +1,0 à +2,5 | 90 % |
| **S3** | E_cal(INDEP) > E_cal(INDEP0) — la curation améliore l'ensemble | 0 à +1,5 | 55 % |
| **S4** | E_cal(M32) > V_cal(COURT2) — l'ensemble de 32 recettes réglées bat une seule de ces recettes | 0 à +0,5 | 60 % |
| **S5** | E_cal(INDEP) > V_cal(INDEP) — l'ensemble bat son propre réseau livré | +2 à +4 | 95 % |
| **S6** | E_cal(INDEP0) > V_cal(GRAND) — même sans curation, l'ensemble bat le grand réseau | 0 à +2,0 | 75 % |
| **S7** | \|H_ens(INDEP)\| < 1 point — l'ensemble, que personne ne choisit, ne se surestime pas (ni ne se sous-estime) de plus d'un point | — | 80 % |
| **S8** | H(SYS) > H_ens(INDEP) — le réseau livré par le système se surestime plus que l'ensemble | +3 à +5 | 95 % |
| **S9** | E_vie(INDEP) > V_vie(GRAND) — pendant l'apprentissage, au seuil par défaut, l'ensemble bat le grand réseau | +5 à +10 | 90 % |
| **S10** | E_cal(INDEP) > V_cal(SEUL) — l'ensemble bat le réseau seul de la clé 40 | +1,5 à +3 | 90 % |
| **S11** | EG_cal(INDEP) > 0 — régler son seuil sur l'ancre améliore l'ensemble | +0,2 à +1,0 | 70 % |
| **S12** | E_cal(M32) > E_cal(SYS) — l'ensemble de 32 recettes réglées bat celui du système | 0 à +0,6 | 50 % |
| **S13** | E_cal(SYS) > V_cal(SYS) — l'ensemble du système bat son réseau livré | +0,3 à +1,2 | 75 % |

Toutes sur la vérité à seuil égal (réglé sur l'ancre), sauf S7 et S8 (surestimation), S9 (sur la vie, au seuil par
défaut) et S11 (ce que rapporte le réglage).

**Deux lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « à seuil égal, l'ensemble de réseaux indépendants est la meilleure capacité mesurée »** : E_cal(INDEP) bat
  V_cal(GRAND), V_cal(COURT2), V_cal(SYS), V_cal(SEUL) et E_cal(M32), chacun à p < 0,025 unilatéral. Je prédis que oui
  (65 %).
- **L2 — « la curation sert l'ensemble, la sélection le dessert »** : E_cal(INDEP) bat E_cal(INDEP0) et E_cal(SYS),
  chacun à p < 0,025 unilatéral. Je prédis que non (55 %) — à cause de la curation.

**Ce que chaque issue voudra dire.**

- Si L1 tient : à seuil égal, au même coût d'usage et au même calcul d'entraînement, la meilleure capacité mesurée par
  nos clés serait un ensemble de réseaux indépendants — une population sans sélection —, le premier gain net et
  pré-enregistré d'une forme de population sur toutes les recettes standard essayées (sur ce monde fixe, ce perceptron).
- Si P1 tient mais pas P2 : l'ensemble bat le grand réseau, mais un ensemble de recettes réglées identiques en fait
  autant : c'est l'ensemble qui compte, pas la diversité des réglages.
- Si S3 tient : la curation, l'une de nos règles, sert l'ensemble ; si S2 tient : la sélection le dessert.
- Si S7 tient : l'ensemble, que personne ne choisit, reste honnête sur lui-même.
- Si P4 tient : la part du gain de la clé 40 qui survivait au meilleur seuil face au réseau seul survit aussi à seuil
  égal.
- Si P1 est démentie : l'avantage de l'ensemble vu à la clé 42 tenait à son seuil, comme le gain du système.

## La règle de décision

Toutes les comparaisons sont appariées par graine, sur les 48 graines. Pour chaque test, l'écart par graine
d = A − B (pour S7 et S11, la valeur A elle-même), le t de Student et la valeur p unilatérale exacte à n − 1 = 47
degrés de liberté.

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ; *indécidable*
  sinon.
- **Tests secondaires (S1–S6, S8–S13)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ;
  *indécidable* sinon. Les p ajustés par Holm sur les 17 tests sont rapportés pour information.
- **Équivalence (S7)** : deux tests unilatéraux, bornes ±0,01 (un point), chacun à 0,025. *Confirmée* si les deux
  rejettent ; *démentie* si la valeur dépasse significativement une borne ; *indécidable* sinon.
- Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
- Plus de deux graines manquantes pour un bras : les tests qui l'utilisent sont *incomplets*.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %.

Le script `verdict43.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle, et calcule les
deux lectures. Il calcule le poids du chat à partir des proportions du monde, comme le moteur ; il diffère de
0,9 / 1,87 de 6·10⁻¹⁷, ce qui ne change aucun résultat (relevé à la clé 42).

## Ce qui a tourné avant ce texte

- `identite43.py` : le moteur de la clé 43 refait au bit près des courses publiées des clés 41 et 42 — la recherche
  aléatoire (graine 48), la meilleure de 32 (graine 66) et le réseau court réglé (graine 50) de la clé 42, entiers ; les
  trois premières générations du système (graine 78) de la clé 42 ; le réseau seul de la clé 40 (graine 30), entier,
  tel que la clé 41 l'a publié : 7 325 valeurs, aucun écart (`identite43*.json`). Le front et la note de l'ancre
  de l'ensemble, nouveaux, s'ajoutent aux sorties sans rien changer.
- `essai43.py` : deux générations d'INDEP0 et d'INDEP (avec les mesures finales) sur la graine 201, hors plan, pour
  vérifier que le bras et les mesures nouvelles tournent ; seules les durées et la forme des sorties ont été affichées.
- `controle43.py --ecrire` puis `controle43.py` : les valeurs attendues du contrôle d'environnement, sur la graine 200
  (hors plan), écrites dans un fichier sans être affichées, puis le contrôle lui-même (« bon » quatre fois).
- `verdict43.py` a été essayé sur des données de synthèse construites à partir des courses de la clé 42 (bras et
  graines renommés, mesures de l'ensemble simulées) — aucune donnée de la clé 43. `ana43.py` (l'analyse descriptive) a
  été écrit avant les courses.
- Rien du plan.

## Où et comment

Les tranches (E1 à E16 : populations et leurs ensembles ; L1 à L6 : grand réseau ; C1 à C4 : réseau seul et réseau
court) tournent dans des sessions cloud que je lance depuis l'ordinateur de Lucas, à sa demande, sur ses crédits ;
chacune exécute une tranche telle qu'écrite (`cle43/CLAUDE.md`) et pousse ses résultats sur sa branche du dépôt public
de la clé 40 (https://github.com/Lucas2a63/cle40), où la clé 43 occupe le dossier `cle43/` ; le bac à sable ne fait que
lire ce dépôt. Rien n'agit hors de ce cadre : pas d'agent autonome, pas d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Une seule architecture (le perceptron), un seul monde (fixe), une seule base d'images, des réseaux en numpy, sans
  augmentation de données, en descente de gradient simple.
- Les réseaux de la recherche aléatoire reçoivent chacun un trente-deuxième du calcul, avec des réglages tirés au
  hasard dans le génome des clés 36 à 41 et un taux constant ; les réglages des recettes viennent des balayages des
  clés 40 et 41, sur une seule graine.
- Le seuil de l'ensemble, un biais sur le logarithme de sa probabilité moyenne du chat, est un choix ; le seuil réglé
  repose sur 2 000 images, dont des répétitions, qui ont aussi servi, pour les populations, au choix du réseau livré.
- L'ensemble coûte à l'usage autant que le grand réseau, 32 fois plus qu'un petit réseau seul ; le calcul
  d'entraînement est égal, le réglage (les balayages) n'est pas compté.
- Les interactions entre ingrédients ne sont pas testées.
