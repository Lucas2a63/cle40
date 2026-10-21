# Clé 45 — Choisir l'ensemble à la fin

Pré-enregistrement, écrit le 21 octobre 2026 vers 19 h 45 (heure de Paris), avant toute course du plan.
Projet Singularité (Lucas).

## La question

À la clé 44, une sélection qui classe à chaque génération les 32 réseaux d'une population selon leur apport à
l'ensemble faisait, à seuil égal, un ensemble moins bon que ne pas sélectionner du tout (−1,06 point en copiant les
poids, −0,94 en ne copiant que les réglages) ; la meilleure population, lue en ensemble, restait la recherche
aléatoire (0,572). Une lecture, non testée : une sélection lue à chaque génération serait myope. Cette clé essaie une
sélection qui ne touche pas à l'apprentissage : **une fois les 120 générations finies, choisir, sur l'ancre, quels
réseaux de la recherche aléatoire entrent dans l'ensemble, et avec quel poids** — la « sélection d'ensemble », une
technique classique (Caruana et coll., 2004). Deux questions, prises ensemble :

1. **Une sélection faite une seule fois, à la fin, sert-elle l'ensemble** — fait-elle mieux que la moyenne des 32
   réseaux, et mieux qu'autant de réseaux tirés au hasard ?
2. **Les réseaux qui servent l'ensemble sont-ils les meilleurs pris un à un ?**

## Ce que les clés 43 et 44 montrent déjà (lu avant ce texte, sur les courses publiées, graines 100 à 147)

- À seuil égal, l'ensemble des 32 réseaux de la recherche aléatoire finit à 0,572 ; son réseau livré (le mieux noté
  par l'ancre) à 0,539 ; le grand réseau de la clé 40 à 0,550 ; l'ensemble de 32 recettes réglées à 0,552. Au seuil
  par défaut, les réseaux de la recherche aléatoire valent 0,480 en moyenne.
- Ses réglages sont tirés au hasard une fois pour toutes : taux de 0,02 à 0,2, décroissance de 10⁻⁵ à 10⁻³ (moyennes
  géométriques 0,063 et 1,0·10⁻⁴).
- Sa diversité (ambiguïté sur le juge, au passage 119) vaut 0,098 ; elle monte tout au long de la vie.
- À la clé 44, la diversité ne suffisait pas à faire l'ensemble : une population au moins à peu près aussi diverse que
  la recherche aléatoire, aux réseaux moins bons en score de Brier, faisait un ensemble un point moins bon (après
  coup).

## Le montage

Tout se passe sur le **perceptron** (3 072 → 128 → 10), en **32×32**, dans le monde des clés 40 à 44 : CIFAR-10, le
chat fatal (3 % du monde, ×30 pour le juge), 500 images par génération tirées avec remise dans les 35 000 images du
flux, un tampon de 5 000, l'ancre de 200 images de la réserve dont 40 chats (réutilisées une fois la réserve épuisée),
le juge sur les 10 000 images de test, 120 générations, monde fixe. Le moteur (`echelle45.py`) est celui de la
clé 44, plus le choix final et le bras INDEPC ; pour les bras des clés 41 à 44, il refait leurs courses au bit près.

**Les graines** sont celles des clés 43 et 44 : **100 à 147**.

**Le bras** : **INDEPC**, la recherche aléatoire (avec curation, sans sélection) — refaite : **elle doit redonner au
bit près toutes les valeurs de ses courses publiées de la clé 43** —, suivie du **choix final**. Rien n'est
réentraîné : les règles choisissent parmi les 32 réseaux finaux de chaque course. Le grand réseau de la clé 40 est
repris de la clé 43, sur les mêmes graines (ses empreintes sont vérifiées par `verdict45.py`).

**Le critère du choix** (fixé ici, calculé par `choix_final` dans `echelle45.py`) : sur les **ancres réunies des dix
dernières générations** (les 2 000 images qui servent déjà à régler le seuil, dont 400 chats, certaines répétées, et
leurs poids, ceux de la note de sélection), la log-vraisemblance pondérée de la bonne classe sous la probabilité
moyenne de l'ensemble choisi — le critère de l'apport à l'ensemble de la clé 44. Le juge n'entre jamais dans le choix.

**Les règles** (l'ensemble d'une règle : la moyenne des probabilités de ses réseaux, chaque réseau compté autant de
fois que sa multiplicité) :

| Règle | Ce que c'est |
|---|---|
| **TOUS** | Les 32 réseaux, à poids égaux : l'ensemble de la recherche aléatoire des clés 43 et 44 (au bit près). |
| **GLOUTON** | La sélection avant, avec remise (Caruana) : partir de rien ; à chaque étape, ajouter le réseau qui maximise le critère de l'ensemble ainsi formé (un réseau peut entrer plusieurs fois : son poids est son nombre d'entrées) ; 64 étapes ; garder l'ensemble de l'étape où le critère est le plus haut (la première, à égalité). |
| **ELAGAGE** | Partir des 32 et retirer, tant que le critère monte strictement, le réseau dont le retrait le fait le plus monter. |
| **MEILLEURS8** | Les 8 réseaux de meilleur critère pris seuls, à poids égaux. |
| **MEILLEURS16** | Les 16 réseaux de meilleur critère pris seuls, à poids égaux. |
| **TETE** | Les d réseaux de meilleur critère pris seuls, à poids égaux, d étant le nombre de réseaux distincts de GLOUTON. |
| **HASARD** | d réseaux tirés au hasard (générateur 20 000 + graine, sans lien avec la course), à poids égaux : le témoin de GLOUTON à coût d'usage égal. |

À égalité de critère, le plus petit indice de réseau l'emporte.

**Le seuil réglé sur l'ancre**, pour l'ensemble de chaque règle, est celui des clés 42 à 44 : la médiane basse des
biais du chat qui maximisent la note de l'ancre sur les mêmes ancres réunies ; la vérité ×30 du juge à ce biais,
V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87. Les mêmes ancres servent donc au choix et au réglage du seuil.

**Le calcul** : 384 000 pas de gradient d'un petit réseau sur 64 images par course, comme aux clés 43 et 44 ; le
choix final ne coûte qu'un passage des ancres réunies et du juge. 48 courses ; environ 25 heures de calcul d'un cœur.

**Les mesures** (par course et par règle) :
- **E_cal** : la vérité ×30 du juge de l'ensemble de la règle, au seuil réglé sur l'ancre.
- **E_fin** : la même au biais 0 (le seuil par défaut).
- **H0** : la note de l'ancre de l'ensemble au biais 0 (sur les ancres réunies) moins sa vérité au biais 0.
- **NB** : le nombre de réseaux distincts — le coût d'usage, en petits réseaux.
- Pour le grand réseau : **V_cal**, au seuil réglé sur l'ancre de son propre front.

Rapportés sans prédiction : E_max (au meilleur seuil du juge, un diagnostic) ; la diversité de l'ensemble de chaque
règle ; le rappel du chat et la précision ordinaire au seuil réglé ; l'ensemble à rappel du chat égal ; ce que
choisit la sélection gloutonne (étape retenue, poids du réseau le plus lourd, réglages et vérité propre des réseaux
retenus, présence du réseau livré) ; le lien, réseau par réseau, entre le critère pris seul et la vérité propre ; ce
que retire l'élagage ; les comparaisons avec le réseau court réglé et le réseau seul de la clé 43 ; les durées.

## Les prédictions

Mon raisonnement. Les réseaux de la recherche aléatoire sont de qualité très inégale — leurs réglages sont tirés au
hasard, et la moyenne des 32 donne à chacun le même poids. Une sélection qui choisit, sur l'ancre, les réseaux qui
servent l'ensemble devrait écarter ou alléger les plus faibles sans perdre la diversité utile : c'est l'argument de
la sélection d'ensemble, et à la clé 44, la qualité des réseaux comptait autant que leur diversité. Mais l'ancre est
petite (2 000 images, 400 chats, certaines répétées) : une sélection gloutonne parmi 32 réseaux, en 64 étapes, peut
s'ajuster à son bruit, et la même ancre règle ensuite le seuil. J'attends donc un gain faible ou nul sur l'ensemble
des 32 (P1, à pile ou face) ; un gain net sur autant de réseaux tirés au hasard (P2) ; que garder les seuls meilleurs
réseaux pris un à un fasse moins bien que tous (P3, avec 8) et moins bien qu'un choix qui regarde l'ensemble, à
taille égale (P4) ; et que l'ancre surestime l'ensemble choisi sur elle (S3).

**Principales** (corrigées ensemble, voir la règle) — graines 100 à 147 :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | E_cal(GLOUTON) > E_cal(TOUS) — à seuil égal, le choix glouton fait un meilleur ensemble que les 32 | 0 à +0,4 | 50 % |
| **P2** | E_cal(GLOUTON) > E_cal(HASARD) — il fait mieux qu'autant de réseaux tirés au hasard | +0,2 à +1,0 | 75 % |
| **P3** | E_cal(TOUS) > E_cal(MEILLEURS8) — les 32 font mieux que les 8 meilleurs pris un à un | +0,2 à +0,8 | 65 % |
| **P4** | E_cal(GLOUTON) > E_cal(TETE) — le choix glouton fait mieux qu'autant de réseaux choisis un à un | 0 à +0,4 | 55 % |

**Secondaires** :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | E_cal(GLOUTON) > V_cal(GRAND) — l'ensemble choisi bat le grand réseau | +1,8 à +2,6 | 95 % |
| **S2** | E_cal(ELAGAGE) > E_cal(TOUS) — l'élagage fait un meilleur ensemble que les 32 | −0,1 à +0,3 | 45 % |
| **S3** | H0(GLOUTON) > H0(TOUS) — l'ancre surestime l'ensemble choisi sur elle plus que celui des 32 | +0,2 à +1,5 | 80 % |
| **S4** | E_cal(MEILLEURS16) > E_cal(MEILLEURS8) — les 16 meilleurs font mieux que les 8 meilleurs | +0,1 à +0,5 | 70 % |
| **S5** | E_cal(TOUS) > E_cal(HASARD) — les 32 font mieux que d réseaux tirés au hasard | +0,1 à +0,8 | 65 % |
| **S6** | E_fin(GLOUTON) > E_fin(TOUS) — au seuil par défaut aussi | 0 à +0,4 | 50 % |
| **S7** | NB(GLOUTON) < 16 — le choix glouton retient moins de la moitié des réseaux (test sur 16 − NB) | 6 à 15 réseaux | 60 % |
| **S8** | E_cal(GLOUTON) > E_cal(MEILLEURS16) — le choix glouton fait mieux que les 16 meilleurs | 0 à +0,5 | 60 % |
| **S9** | E_cal(TOUS) > E_cal(MEILLEURS16) — les 32 font mieux que les 16 meilleurs | 0 à +0,4 | 50 % |

Toutes sur la vérité à seuil égal (réglé sur l'ancre), sauf S3 (surestimation), S6 (seuil par défaut) et S7 (nombre de
réseaux). Les écarts de vérité sont en points (centièmes).

**Deux lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « une sélection faite une seule fois, à la fin, sert l'ensemble »** : P1 et P2 confirmées. Je prédis que non
  (55 %).
- **L2 — « les meilleurs réseaux pris un à un ne font pas le meilleur ensemble »** : P3 et P4 confirmées. Je prédis
  que non (60 %).

**Ce que chaque issue voudra dire.**

- Si L1 tient : une sélection, faite une fois sur les réseaux finis et non à chaque génération, sert l'ensemble — la
  première fois, dans ce projet, qu'une sélection ajoute à la meilleure capacité mesurée à seuil égal (sur ce monde
  fixe, ce perceptron) ; et la lecture de la myopie y gagne un appui, sans être prouvée.
- Si P2 tient et pas P1 : choisir vaut mieux que le hasard à taille égale, sans battre la moyenne de tous — le choix
  ne sert qu'à économiser des réseaux à l'usage.
- Si P1 est démentie : même faite une fois, à la fin, une sélection sur l'ancre coûte à l'ensemble ; l'ancre est trop
  petite pour choisir.
- Si L2 tient : les réseaux qui servent l'ensemble ne sont pas les meilleurs pris un à un.

## La règle de décision

Toutes les comparaisons sont appariées par graine, sur les 48 graines. Pour chaque test, l'écart par graine
d = A − B (pour S7, 16 − NB lui-même), le t de Student et la valeur p unilatérale exacte à n − 1 = 47 degrés de
liberté.

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ; *indécidable*
  sinon.
- **Tests secondaires (S1–S9)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ; *indécidable*
  sinon. Les p ajustés par Holm sur les 13 tests sont rapportés pour information.
- Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
- Plus de deux graines manquantes : les tests sont *incomplets*.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %.
- **Identité** : les 48 courses INDEPC doivent redonner au bit près toutes les valeurs de leurs courses INDEP publiées
  de la clé 43 (hors durée et tâche), et la règle TOUS le front, le front de l'ancre et la diversité de leur
  ensemble des 32. Une seule valeur différente rend la clé invalide : je le dirai, et aucun verdict ne sera présenté
  comme tel.

Le script `verdict45.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle, vérifie
l'identité et les empreintes des courses reprises de la clé 43, et calcule les deux lectures.

## Ce qui a tourné avant ce texte

- `identite45.py` : le moteur de la clé 45 refait au bit près la course publiée de la recherche aléatoire de la
  clé 43, graine 100, entière : 3 337 valeurs, aucun écart ; sa règle TOUS redonne au bit près le front, le front de
  l'ancre et la diversité de son ensemble des 32 (`identite45.json`). Le choix final s'ajoute aux sorties sans rien
  changer. Cette course est celle du plan pour la graine 100 : elle a tourné entière ici, mais rien de son choix final
  n'a été affiché ni enregistré — seulement les comptes de l'identité.
- Un essai de deux générations du bras INDEPC sur la graine 201, hors plan, pour vérifier que le choix final tourne :
  seuls la durée, la forme des sorties et, pour ces deux générations hors plan, le nombre de réseaux retenus par
  chaque règle, son critère et le début du chemin du choix glouton ont été affichés.
- `controle45.py --ecrire` puis `controle45.py` : les valeurs attendues du contrôle d'environnement, sur la graine 200
  (hors plan), écrites dans un fichier sans être affichées, puis le contrôle lui-même (« bon » deux fois).
- `verdict45.py` et `ana45.py` ont été essayés sur des données de synthèse construites à partir des courses de la
  recherche aléatoire de la clé 44 (règles de choix inventées, au hasard) — aucune donnée de la clé 45. `ana45.py`,
  l'analyse descriptive, a été écrit avant les courses et figure dans le manifeste.
- Rien d'autre du plan.

## Où et comment

Les tranches (F1 à F12, 4 courses chacune) tournent dans des sessions cloud que je lance depuis l'ordinateur de Lucas,
à sa demande, sur ses crédits ; chacune exécute une tranche telle qu'écrite (`cle45/CLAUDE.md`) et pousse ses
résultats sur sa branche du dépôt public de la clé 40 (https://github.com/Lucas2a63/cle40), où la clé 45 occupe le
dossier `cle45/` ; le bac à sable ne fait que lire ce dépôt. Rien n'agit hors de ce cadre : pas d'agent autonome, pas
d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Une seule architecture (le perceptron), un seul monde (fixe), une seule base d'images, des réseaux en numpy, sans
  augmentation de données, en descente de gradient simple.
- Une seule population (la recherche aléatoire) et une seule bibliothèque de 32 réseaux par course ; la sélection
  d'ensemble est d'ordinaire faite sur des bibliothèques plus grandes et plus variées.
- Une seule ancre, petite, sert à choisir et à régler le seuil ; la sélection gloutonne n'a ni initialisation triée
  ni ensachage, qui limitent l'ajustement au bruit dans la littérature.
- Un seul critère de choix (la log-vraisemblance pondérée) ; un autre (la note, l'exactitude) pourrait faire
  autrement.
- Le grand réseau est repris de la clé 43 ; l'ensemble coûte à l'usage autant de petits réseaux qu'il en retient.
