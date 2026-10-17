# Clé 40 — L'échelle : le même monde, en pleine résolution

Pré-enregistrement, écrit le 17 octobre 2026 à 3 h 19 (heure de Paris), avant toute course du plan et avant le
balayage de réglage.
Projet Singularité (Lucas).

## La question

La clé 39 a mis le système entier — 32 réseaux, l'ancre visée sur le fatal, la sélection, la bascule — face
à la recette standard, à calcul d'entraînement égal, sur CIFAR-10 réduit en 16×16. Réponse : il ne bat
nettement la meilleure recette standard sur aucune architecture. Sur le convolutif, quatre recettes se
tenaient dans un point ; sur le perceptron, le système battait les deux recettes à calcul égal, mais un réseau
seul 32 fois moins coûteux faisait presque aussi bien. La question laissée ouverte : **à plus grande échelle,
l'écart change-t-il, et dans quel sens ?** Cette clé fait le premier pas : le même monde, vu en pleine
résolution (32×32, les images d'origine), avec des réseaux quatre fois plus coûteux par image et le même nombre
de pas. Tout le reste est inchangé, et chaque course de la clé 40 voit, à graine égale, les mêmes images que
la course de la clé 39 — ce qui permet de lire, graine par graine, ce que l'échelle change.

## Le montage

**Le monde, les données, l'ancre et le juge** sont ceux des clés 36 à 39, sauf la résolution : CIFAR-10 en
32×32 (au lieu de 16×16) ; le chat est fatal (3 % du monde, ×30 pour le juge) ; le monde envoie 500 images
neuves par génération dans un tampon de 5 000 ; l'ancre tire à chaque génération 200 images neuves de la
réserve, dont 40 chats, pesées au coût ; le juge lit les 10 000 images de test ; 120 générations ; monde fixe.
Les images 16×16 des clés 36 à 39 sont exactement la moyenne arrondie de chaque bloc 2×2 des images 32×32,
dans le même ordre (vérifié sur les 60 000 images) ; le découpage flux / réserve et les tirages du monde sont
les mêmes : à graine égale, une course de la clé 40 reçoit les mêmes images du flux et les mêmes ancres que la
course de la clé 39, et le même juge. Les tirages des réseaux (initialisation, génomes, sélection, lots)
diffèrent, puisque les réseaux n'ont pas la même taille.

**Les réseaux.** Convolutif : conv 3×3 de pas 2 (32×32×3 → 16×16×16) → conv 3×3 (16×16×32) → moyenne 2×2
(8×8×32) → 2 048 → 10, tanh — les mêmes couches qu'à la clé 39, sur des cartes quatre fois plus grandes.
Perceptron : 3 072 → 128 → 10. GRAND : convolutif à 96 et 186 filtres, perceptron à 4 096 neurones, 32 fois
plus coûteux par image (convolutif : 31,98 fois en inférence, 32,74 à l'entraînement ; perceptron : 32,0). Un
réseau de la population coûte environ quatre fois plus par image qu'à la clé 39.

**L'unité de calcul** est le pas : un pas de gradient d'un réseau sur un lot de 64 images. Le système entier
dépense 32 × 100 = 3 200 pas par génération (384 000 en tout). Chaque bras dépense autant de calcul, sauf
COURT ; GRAND le dépense en 100 pas par génération, chacun 32 fois plus coûteux. L'évaluation par l'ancre
coûte environ 1 à 1,5 % du calcul du système ; le juge est un instrument de mesure, hors du compte.

**Les bras** sont ceux de la clé 39 :

| Bras | Ce que c'est | Pas par génération |
|---|---|---|
| **SYS** | Le système entier : 32 réseaux ; l'ancre les note ; à chaque génération, le quart le moins bien noté est remplacé par des copies mutées du quart le mieux noté (poids copiés ; génome muté : taux d'apprentissage, décroissance, curation) ; ×30 dans la perte à la génération 10 pour le convolutif, dès 0 pour le perceptron. Ses bornes de génome sont celles de la clé 36 (taux de départ tirés entre 0,02 et 0,2) : le système n'est pas re-réglé pour l'échelle — sa sélection doit s'y ajuster seule. Livre le réseau le mieux noté par l'ancre à la dernière génération. | 32 × 100 |
| **INDEP** | La même population sans sélection (une recherche aléatoire de réglages) ; livre aussi le mieux noté. | 32 × 100 |
| **SEUL** | La recette standard : un réseau seul de la même architecture ; ×30 dès le départ ; tirage uniforme dans le tampon ; taux en cosinus, de lr₀ à 0 ; décroissance des poids 1e-4. | 3 200 |
| **SEUL10** | SEUL avec la bascule à la génération 10 (convolutif seulement). | 3 200 |
| **GRAND** | La recette standard avec un réseau 32 fois plus coûteux par image. | 100 |
| **COURT** | SEUL avec 100 pas par génération : le calcul d'un seul réseau de la population. | 100 |

**Le réglage de la recette standard** (`choix40.py`, `plan40.py`), fixé ici avant tout balayage. Le taux de
départ lr₀ de SEUL, COURT et GRAND est choisi, pour chaque architecture, par un balayage sur la graine 99 (hors
plan), avec l'ancre pour seule note — la moyenne de ses notes aux dix dernières générations (2 000 images) —,
jamais le juge. SEUL10 prend le taux de SEUL. Grilles de départ (multiples de LR0 = 0,0632), centrées un peu
sous les taux retenus à la clé 39, parce que des cartes plus grandes et une entrée plus large font des pas
effectifs plus grands :

- SEUL, convolutif : {1/256, 1/128, 1/64, 1/32, 1/16, 1/8}
- GRAND et COURT, convolutif : {1/8, 1/4, 1/2, 1, 2}
- SEUL, perceptron : {1/1024, 1/512, 1/256, 1/128, 1/64, 1/32}
- GRAND, perceptron : {1/128, 1/64, 1/32, 1/16, 1/8}
- COURT, perceptron : {1/64, 1/32, 1/16, 1/8, 1/4}

Le taux retenu est celui de la meilleure note. S'il est au bord de la grille, la grille est prolongée de deux
facteurs 2 de ce côté (deux courses de plus), au plus deux fois par bras. Leçon de la clé 39, où le meilleur taux
du réseau seul était resté au bord après trois prolongements d'un facteur 2 : des grilles plus larges dès le
départ, et des prolongements plus longs. Le balayage tourne après ce texte (tranches B1 à B3) ; le script
n'affiche que les taux retenus ; je ne lirai les notes qu'après le verdict. Il est hors du compte de la course.

**Les graines** : 0 à 11 pour le convolutif ; 0 à 23 pour le perceptron — les mêmes qu'à la clé 39. 192 courses
dans le plan (72 convolutives, 120 du perceptron), plus le balayage (32 courses de départ). D'après le pilote,
le plan représente environ 160 heures de calcul d'un cœur du bac à sable — dont 77 pour les 24 populations du
convolutif, environ 3 h 15 chacune —, et le balayage une vingtaine ; un cœur du cloud est de 1,3 à 2 fois plus
lent.

**Les mesures** sont celles de la clé 39. Le *réseau livré* est le mieux noté par l'ancre pour SYS et INDEP,
le réseau lui-même pour les autres bras.

- **V_fin** : la vérité ×30 du juge pour le réseau livré, à la génération 119.
- **V_vie** : la même, en moyenne sur les 16 passages du juge 0, 8, …, 112, 119.
- **H** : la note que l'ancre donne au réseau livré, moins sa vérité selon le juge, en moyenne sur les 11
  passages 40, 48, …, 112, 119.
- **E_fin** : la vérité ×30 de l'ensemble des 32 réseaux (moyenne des probabilités), génération 119.
- Mesures finales (banc de stress, sonde, dimension effective) : rapportées sans prédiction. En 32×32, le
  flou et la pixellisation du banc de stress sont mis à l'échelle de l'image (flou deux fois plus large en
  pixels ; pixellisation de 32 à 16 côtés) ; la sonde et la dimension effective du convolutif se lisent, comme
  aux clés 36 à 39, sur des traits ramenés à 4×4×C2 (une moyenne 2×2 de plus).
- **Les écarts d'échelle** : pour une graine, la mesure (ou l'écart entre deux bras) en 32×32 moins la même en
  16×16, à la clé 39 (`reference39.json`, extrait des courses publiées par `reference39.py`).

## Les prédictions

Mon raisonnement. En 32×32, chaque image porte quatre fois plus de pixels, et chaque réseau quatre fois plus
de calcul par image. Les lois d'échelle observées en apprentissage profond disent qu'à calcul fixe, quand
l'information est riche, mieux vaut un modèle plus grand entraîné moins longtemps : le grand réseau, qui ne
s'était pas distingué en 16×16 (+0,47 point, indécidable), devrait prendre l'avantage sur le convolutif, et
d'autant plus que l'échelle grandit. Le petit réseau seul repasse toujours chaque image du tampon quelque
410 fois : il sur-apprend encore, et le système, qui convertit son calcul en recherche de réglages, devrait
le battre. Sur le perceptron, j'avais parié en 16×16 que le grand réseau battrait le système, et j'avais eu
tort (−0,65 point) : je parie maintenant que ce résultat tient. Le reste reprend, à l'échelle, ce que la
clé 39 a établi.

**Principales** (corrigées ensemble, voir la règle) :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | Convolutif : V_fin(GRAND) > V_fin(SYS) — en 32×32, le grand réseau bat le système | +1 à +4 points | 65 % |
| **P2** | Convolutif : [V_fin(GRAND) − V_fin(SYS)] en 32×32 > le même écart en 16×16, graine par graine — l'avance du grand réseau grandit avec l'échelle | +0,5 à +3 | 60 % |
| **P3** | Perceptron : V_fin(SYS) > V_fin(GRAND) — le système bat encore le grand réseau | +0,3 à +2 | 60 % |
| **P4** | Perceptron : V_fin(SYS) > V_fin(SEUL) — le système bat le réseau seul qui reçoit tout le calcul, même réglé sur une grille élargie | +2 à +10 | 75 % |

**Secondaires** :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | Convolutif : V_fin(SYS) > V_fin(INDEP) — la sélection bat la recherche aléatoire | +1 à +4 | 85 % |
| **S2** | Perceptron : idem | +1 à +4 | 85 % |
| **S3** | Convolutif : V_fin(SYS) > V_fin(COURT) — la population vaut ses 32 réseaux | +0,3 à +2 | 65 % |
| **S4** | Perceptron : idem | +0,2 à +2 | 50 % |
| **S5** | Convolutif : V_fin(SEUL10) > V_fin(SEUL) — la bascule améliore la recette standard | +0,3 à +2 | 65 % |
| **S6** | Convolutif : V_fin(SYS) > V_fin(SEUL) | +0,3 à +2 | 55 % |
| **S7** | Convolutif : V_fin(GRAND) > V_fin(SEUL) — à calcul égal, le grand réseau bat le petit réseau entraîné longtemps | +1 à +4 | 75 % |
| **S8** | Convolutif : V_vie(SEUL) > V_vie(SYS) — la recette standard protège mieux pendant l'apprentissage (le prix des dix générations d'attente) | +1 à +4 | 85 % |
| **S9** | Perceptron : V_vie(SYS) > V_vie(SEUL) — le système protège mieux que le réseau seul | +3 à +15 | 75 % |
| **S10** | Convolutif : H(SYS) > H(SEUL) — le système se surestime, le réseau seul non | +1 à +4 | 85 % |
| **S11** | Perceptron : idem | +1 à +4 | 80 % |
| **S12** | Convolutif : E_fin(INDEP) > V_fin(INDEP) — l'ensemble des 32 indépendants bat le meilleur d'entre eux | +0,5 à +3 | 70 % |
| **S13** | Perceptron : idem | +2 à +6 | 85 % |
| **S14** | Convolutif : \|E_fin(SYS) − V_fin(SYS)\| < 1 point — l'ensemble du système n'ajoute pas plus d'un point | — | 70 % |
| **S15** | Convolutif : V_fin(SYS) en 32×32 > V_fin(SYS) en 16×16, graine par graine — l'échelle rend le système plus vrai | +1 à +5 | 75 % |

**Comparaisons rapportées sans prédiction** (donc sans verdict) : la meilleure des recettes standard contre le
système ; l'écart d'échelle de chaque bras et de chaque comparaison de la clé 39 ; les ensembles ; le banc de
stress, la sonde, la dimension effective ; le rappel du chat, la précision ordinaire et les fausses alertes ;
le taux d'apprentissage découvert par la population ; les taux retenus par le balayage ; la durée des courses.

**Ce que chaque issue voudra dire.**

- *Gain de capacité de la population* — lecture écrite d'avance, pour chaque architecture : le système bat
  **chacune** de ses recettes standard (convolutif : SEUL, SEUL10, GRAND, COURT ; perceptron : SEUL, GRAND,
  COURT), chacune à p < 0,025 unilatéral. Il faut les gagner toutes, réseau court compris : c'est la leçon de la
  clé 39, où battre les recettes à calcul égal ne suffisait pas quand un réseau 32 fois moins coûteux faisait
  presque aussi bien. Je prédis que ce n'est le cas sur aucune architecture.
- Si P1 et P2 sont confirmées : à l'échelle, la taille devient le levier — un réseau plus grand tire mieux parti
  du calcul que la population, et de plus en plus quand l'information par image grandit. Pour aller vers plus
  de capacité, il faudra de plus grands réseaux ; nos règles serviront à les garder vrais.
- Si P1 est démentie (le système bat le grand convolutif) et que la lecture « gain de capacité » tient : ce
  serait la première fois que nos règles apportent de la capacité à calcul égal — à éprouver plus grand encore.
- Si P2 est démentie : l'échelle sert la population plus que la taille.

## La règle de décision

Toutes les comparaisons sont appariées par graine. Pour chaque test, l'écart par graine d (A − B, ou, pour un
test d'échelle, l'écart en 32×32 moins le même en 16×16), le t de Student et la valeur p unilatérale exacte à
n − 1 degrés de liberté (n = 12 pour le convolutif, 24 pour le perceptron).

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ;
  *indécidable* sinon.
- **Tests secondaires (S1–S13, S15)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ;
  *indécidable* sinon. Les p ajustés par Holm sur les 19 tests sont rapportés pour information.
- **Équivalence (S14)** : deux tests unilatéraux, bornes ±0,01, chacun à 0,025. *Confirmée* si les deux
  rejettent ; *démentie* si l'écart dépasse significativement une borne ; *indécidable* sinon.
- Plus de deux graines manquantes pour un bras : les tests qui l'utilisent sont *incomplets*.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %.

Le script `verdict40.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle, et calcule
la lecture « gain de capacité ».

## Ce qui a tourné avant ce texte

- `identite40.py` : le moteur de la clé 40, en 16×16, refait au bit près sept courses publiées de la clé 39 —
  quatre entières (COURT convolutif et perceptron, SEUL et GRAND perceptron) et les trois premières
  générations de trois populations (SYS convolutif et perceptron, INDEP convolutif) : 4 485 valeurs, écart 0
  (`identite40.json`).
- `gradients40.py` : la rétropropagation des quatre réseaux en 32×32 contre des différences finies, en double
  précision (écart relatif sous 10⁻⁸).
- `pilote40.py` : quelques générations de chaque bras sur la graine 98 (hors plan, hors balayage), pour la
  durée et la mémoire ; seules les durées et la mémoire ont été affichées.
- `controle40.py --ecrire` : les valeurs attendues du contrôle d'environnement (quatre débuts de course).
- Rien du balayage.

## Où et comment

Le balayage (B1 à B3), les populations (P1 à P8) et, une fois les taux fixés, les réseaux seuls (R1 à R8)
tournent dans des sessions cloud que je lance depuis l'ordinateur de Lucas, à sa demande, sur ses crédits ;
chacune exécute une tranche telle qu'écrite (`CLAUDE.md`) et pousse ses résultats sur sa branche du dépôt
public ; le bac à sable ne fait que lire ce dépôt, et fait tourner les prolongements éventuels du balayage.
Rien n'agit hors de ce cadre : pas d'agent autonome, pas d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- L'échelle ici, c'est quatre fois plus d'information par image et de calcul par pas, sur les mêmes 50 000
  images et le même flux : pas plus d'images, ni des réseaux plus larges en canaux. Un seul monde (fixe), un
  seul niveau de rareté du fatal, de petits réseaux en numpy, sans augmentation de données, en descente de
  gradient simple.
- Le système garde les bornes de génome de la clé 36 alors que la recette standard est re-réglée : c'est voulu
  (on teste l'auto-réglage), mais cela peut le désavantager si ses bornes ne conviennent plus à l'échelle.
- Le moment de peser du système (bascule à 10 pour le convolutif, dès 0 pour le perceptron) a été choisi aux
  clés 36 à 38, en 16×16, sur ces mêmes graines.
- Le calcul est égal à l'entraînement, pas à l'usage : l'ensemble et le grand réseau coûtent 32 fois plus à
  interroger.
- La recette standard est réglée sur une seule graine.
- Les tests d'échelle comparent deux clés qui diffèrent par la résolution et par tout ce qu'elle entraîne
  (taille des cartes, de la tête, tirages des réseaux) : ils disent ce que fait le passage de 16×16 à 32×32,
  pas lequel de ces changements le fait.
