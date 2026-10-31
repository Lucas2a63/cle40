# Clé 48 — La moyenne des poids sur un taux cyclique, et le découpage quand le calcul manque

Pré-enregistrement, écrit le 25 octobre 2026 — commencé vers 2 h 45, figé à 4 h 23 (heure de Paris, CET, UTC+1 : la nuit
du passage à l'heure d'hiver) —, avant toute course du plan. Projet Singularité (Lucas).

## La question

À la clé 47, à calcul d'entraînement égal et à seuil égal, la recette du réseau seul (SEUL10) coupée en deux, quatre ou
huit réseaux, lus en ensemble, a battu le réseau seul de 1,4 à 1,8 point — alors qu'aucun de ces réseaux, pris seul, ne
faisait mieux que lui : un réseau nourri de la moitié du calcul ne se distinguait pas du réseau seul (+0,02 [−0,35 ;
+0,40] en moyenne sur les deux réseaux de K2, sans prédiction). Le réseau seul avait peut-être plus de calcul qu'il n'en
tirait parti. Et, sans prédiction, un seul réseau a recueilli une part du multiple au même calcul et au même coût
d'usage : la moyenne des poids des quatre instantanés d'un entraînement au taux cyclique (la Toile), +0,70 point. Cette
clé pose deux questions :

1. **Un seul réseau recueille-t-il le multiple, au même calcul et au même coût d'usage ?** La moyenne des poids sur un
   taux cyclique, cette fois prédite, sur des graines neuves (quatre cycles, et huit) ; et la distillation à calcul
   total égal — le maître et l'élève se partagent le calcul du réseau seul (à la clé 47, la Photographie en coûtait deux
   fois).
2. **Le découpage gagne-t-il encore quand le calcul manque ?** Le réseau seul, deux réseaux, quatre réseaux, qui se
   partagent le quart, puis le seizième du calcul de SEUL10 — là où un réseau seul ne peut plus se passer d'une part de
   son calcul.

## Ce que la clé 47 montre déjà (lu avant ce texte, sur ses courses publiées ; graines 100 à 123)

À seuil égal (réglé sur l'ancre), en points de vérité ×30 du juge ; le réseau seul SEUL10 à 66,00.

- Tests de la clé 47 : la recette coupée en deux, quatre, huit réseaux bat le réseau seul, +1,44 [+1,12 ; +1,76], +1,84
  [+1,57 ; +2,10], +1,45 [+1,12 ; +1,77] ; coupée en 32, elle perd, −0,56 [−0,94 ; −0,18]. La Photographie (un élève de
  la recette de SEUL10, distillé de l'ensemble K4 : deux fois le calcul) bat le réseau seul, +1,01 [+0,67 ; +1,35], et
  perd contre son maître, −0,82 [−1,10 ; −0,55]. Les quatre instantanés de la Toile sont plus divers que les quatre
  réseaux de K4 (ambiguïté 0,084 contre 0,036). SWA, EMA et la Soupe ne se distinguent pas du réseau seul.
- Sans prédiction (l'analyse descriptive écrite avant les courses, et les comparaisons faites après le verdict) :
  - les réseaux de K2, K4, K8, K32 pris seuls (en moyenne sur les réseaux de chaque ensemble), nourris de 1/2, 1/4, 1/8,
    1/32 du calcul : +0,02 [−0,35 ; +0,40], −0,18 [−0,47 ; +0,11], −0,43 [−0,76 ; −0,11], −1,95 [−2,27 ; −1,63] face au
    réseau seul ; d'un découpage au suivant : −0,20 [−0,42 ; +0,01] (de 1/2 à 1/4), −0,25 [−0,37 ; −0,13] (de 1/4 à
    1/8), −1,51 [−1,61 ; −1,42] (de 1/8 à 1/32) ;
  - ce que l'ensemble gagne sur ses propres réseaux : +1,42 (K2), +2,02 (K4), +1,88 (K8), +1,39 (K32) ; leur ambiguïté
    sur le juge : 0,030, 0,036, 0,033, 0,022 ;
  - la Toile : l'ensemble des quatre instantanés +0,82 [+0,50 ; +1,14] ; **la moyenne de leurs poids (MOY_TOILE) +0,70
    [+0,31 ; +1,09]**, 19 graines sur 24, écart-type par graine 0,92 — l'un d'une trentaine d'écarts au réseau seul lus
    sans prédiction ; au meilleur seuil du juge +0,69 [+0,42 ; +0,97] ; **au seuil par défaut (biais 0) −2,68 [−3,20 ;
    −2,16]**, son biais du chat réglé sur l'ancre valant 2,6 en moyenne contre 1,6 pour le réseau seul ; face à
    l'ensemble des instantanés −0,12 [−0,34 ; +0,10] ; le dernier instantané (le réseau final) −1,44 [−1,80 ; −1,07],
    les quatre instantanés −3,58, −2,44, −1,94, −1,44 ; calculé pour ce texte, la recette coupée en deux la devance de
    0,74 [+0,37 ; +1,12] point, coupée en quatre de 1,14 [+0,82 ; +1,45] ; et, calculés pour ce texte aussi, les
    instantanés pris seuls demandaient un biais du chat plus fort encore que leur moyenne (2,89, 3,23, 3,44, 3,35 en
    moyenne, contre 2,64 pour la moyenne et 1,64 pour le réseau seul) et perdaient davantage au biais 0 (vérités 55,96,
    57,18, 57,88, 58,44, contre 61,24 pour la moyenne et 63,92 pour le réseau seul) ;
  - la moyenne des poids de réseaux sans ancêtre commun s'effondre (de −17,7 à −18,8).

## Le montage

Tout se passe dans le monde des clés 40 à 47 : CIFAR-10 en 32×32, le chat fatal (3 % du monde, ×30 pour le juge), 500
images par génération tirées avec remise dans les 35 000 images du flux, un tampon de 5 000, l'ancre de 200 images de la
réserve dont 40 chats, le juge sur les 10 000 images de test, 120 générations, monde fixe. L'architecture est le
**réseau convolutif de la clé 40** (environ 25 600 paramètres). Le moteur (`echelle48.py`) est celui de la clé 47,
inchangé (seul son texte d'en-tête est réécrit), avec une section nouvelle pour les bras de cette clé, qui ont leurs
points de reprise intégrés.

**Le calcul** : celui de SEUL10, 384 000 pas de gradient d'un petit réseau convolutif sur 64 images (3 200 par
génération). Un réseau qui reçoit 1/n de ce calcul a le taux de la clé 47, **0,0625 · n^0,6 · LR0** (la loi de puissance
qui passe par les deux taux réglés de la clé 40, n = 1 et n = 32), en cosinus sur ses propres pas ; ici n = 1, 2, 4, 8,
16, 32 et 64.

**Partie A — un seul réseau recueille-t-il le multiple ?** Graines neuves, **124 à 147**. Cinq bras :

| Bras | Ce que c'est |
|---|---|
| **SEUL10** | Le réseau seul pesé à la 10e génération, nourri de tout le calcul (le calcul de la clé 46, au bit près). |
| **SEUL_H** | Le même, nourri de la moitié du calcul (1 600 pas par génération, au taux 0,095 · LR0). |
| **PHOTO2** | Le maître, **K2_H** : la recette coupée en deux réseaux qui se partagent la moitié du calcul (800 pas par génération chacun, au taux 0,144 · LR0, celui des réseaux de K4), lus en ensemble. Puis ses probabilités adoucies à la température 4 sur les 35 000 images du flux, et l'élève : un seul réseau nourri de l'autre moitié du calcul, avec le générateur de SEUL10 et de SEUL_H — les poids de départ, les paquets, le monde et l'ancre de SEUL_H, son jumeau sans maître —, sur la perte de la clé 47, w(y)·[½ entropie croisée + ½ · 4² · KL(maître ‖ élève)]. Le maître et l'élève ensemble : le calcul de SEUL10. |
| **TOILE4** | La Toile de la clé 47, au bit près : un seul réseau, le calcul et le générateur de SEUL10, un taux en cosinus qui redémarre quatre fois (quatre cycles de 30 générations, chacun de 0,144 · LR0 à 0), un instantané à la fin de chaque cycle. Mesurés : **MOY_TOILE4**, la moyenne des poids des quatre instantanés ; **MOY_TOILE4_FIN**, celle des deux derniers ; l'ensemble des quatre (TOILE4) ; le dernier (TOILE4.DERNIER) ; chacun. |
| **TOILE8** | La même à huit cycles de 15 générations, chacun de 0,218 · LR0 à 0 : **MOY_TOILE8** (les huit), **MOY_TOILE8_FIN** (les quatre derniers), l'ensemble des huit, le dernier, chacun. |

**Partie B — le découpage quand le calcul manque.** Graines **100 à 123**, celles des clés 46 et 47 : les courses
publiées de la clé 47 (le réseau seul SEUL10, K2, K4, et les réseaux de K8 et K32 pris seuls) servent de référence,
graine par graine, dans le même monde (même moteur, mêmes graines) ; leurs empreintes sont inscrites dans
`verdict48.py`. Six bras :

| Bras | Le calcul | Réseaux | Pas par génération et par réseau | Taux de départ |
|---|---|---|---|---|
| **SEUL_Q** | 1/4 | 1 | 800 | 0,144 · LR0 |
| **K2_Q** | 1/4 | 2 | 400 | 0,218 · LR0 |
| **K4_Q** | 1/4 | 4 | 200 | 0,330 · LR0 |
| **SEUL_S** | 1/16 | 1 | 200 | 0,330 · LR0 |
| **K2_S** | 1/16 | 2 | 100 | 0,500 · LR0 |
| **K4_S** | 1/16 | 4 | 50 | 0,758 · LR0 |

Chaque découpage est la recette de SEUL10 (cosinus, décroissance WD0, sans curation, tirage uniforme dans le tampon, ×30
dans la perte à partir de la 10e génération), en k réseaux indépendants — initialisations et paquets propres —, lus en
ensemble (la moyenne de leurs probabilités). Le réseau seul du quart a la recette d'un réseau de K4 de la clé 47 ; les
réseaux de K2_Q celle des réseaux de K8, ceux de K2_S celle des réseaux de K32 (le taux réglé du réseau court de la clé
40), ceux de K4_Q celle du réseau seul du seizième.

**Le seuil réglé sur l'ancre**, pour chaque ensemble et chaque réseau, est celui des clés 42 à 47 : la médiane basse des
biais du chat qui maximisent la note de l'ancre sur les ancres réunies des dix dernières générations ; la vérité ×30 du
juge à ce biais, V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87.

**Les mesures** (par course) : **E_cal** (un ensemble) et **V_cal** (un réseau seul) : la vérité ×30 du juge au seuil
réglé sur l'ancre — la mesure principale ; E_fin / V_fin au biais 0 ; E_max / V_max au meilleur biais du juge (un
diagnostic) ; **AMB**, l'ambiguïté d'un ensemble sur le juge ; et, nouvelle, **la confiance** sur le juge : l'échelle
des logits (**LOG** : l'écart-type des dix logits d'une image, moyen sur les images), la plus grande probabilité
moyenne, l'entropie, la log-vraisemblance négative de la bonne classe, la probabilité du chat sur les chats. Dérivées,
par graine de la partie B : **GAIN** (un ensemble moins le réseau seul au même calcul : GAIN.K2_47 = K2 − SEUL10 de la
clé 47, GAIN.K2_S = K2_S − SEUL_S), **GMEM** (un ensemble moins la moyenne de ses réseaux pris seuls), **PRIX** (le
réseau seul moins les réseaux de K2 au même calcul total, nourris chacun de la moitié : PRIX.Q, PRIX.S).

**Le coût d'usage** (multiplications par image à l'usage) : un réseau pour SEUL10, SEUL_H, PHOTO2 (l'élève), les
moyennes de poids, SEUL_Q et SEUL_S ; deux pour K2_H, K2_Q, K2_S ; quatre pour TOILE4 (l'ensemble), K4_Q, K4_S ; huit
pour TOILE8 (l'ensemble).

## Les prédictions

Mon raisonnement.

*Partie A.* À la clé 47, un réseau nourri de la moitié du calcul ne se distinguait pas du réseau seul : si la moitié lui
suffit, le reste de son calcul est en trop. La Toile en fait autre chose : elle redémarre le taux, et chaque cycle
laisse un instantané. Ses instantanés restent dans un même bassin — leur moyenne ne s'effondre pas, quand celle de
réseaux sans ancêtre commun perd 18 points —, mais ils sont divers (plus que quatre réseaux indépendants) : la moyenne
de leurs poids tombe vers le centre de la région qu'ils explorent, plus plate (Izmailov et coll., 2018, « SWA » — une
lecture de la littérature, que cette clé ne mesure pas), et vaut bien mieux que chacun d'eux. Le +0,70 de la clé 47 a
été lu parmi une trentaine d'écarts sans prédiction : il a été remarqué parce qu'il était grand, et doit rétrécir sur
des graines neuves ; j'attends de +0,2 à +0,9 (**P1**, 60 %). La moyenne des poids rend presque tout l'ensemble des
instantanés, au coût d'usage d'un réseau : je prédis qu'elle le vaut à 0,5 point près (**S2**, 75 %), et que l'ensemble
des quatre instantanés bat le réseau seul (**S3**, 90 %). Au seuil par défaut, la moyenne perd (**S4**, 97 %) : à la clé
47, ses instantanés pris seuls demandaient un biais du chat plus fort encore et perdaient davantage au biais 0 — la
perte au seuil par défaut vient de l'entraînement cyclique, et la moyenne en rattrape une part ; je n'en ai pas
d'explication. À part : la moyenne des poids d'instantanés qui ne pointent pas dans la même direction devrait avoir des
poids plus petits (la norme d'une moyenne ne dépasse pas la moyenne des normes), donc des logits plus petits que ceux de
ses instantanés — et que ceux du réseau seul, si les instantanés ont à peu près son échelle (**S5**, 70 %) : une
prédiction sur l'échelle des logits, qui ne dit rien du seuil — changer l'échelle des logits ne change aucune décision
au biais 0, et demanderait un biais plus petit, non plus grand. À huit cycles, plus d'instantanés, mais des cycles deux
fois plus courts, qui repartent d'un taux plus haut (0,218 · LR0) : plus de diversité, plus de risque de sortir du
bassin, et des premiers instantanés plus faibles ; je ne sais pas (**S1**, 45 %).

La Photographie à calcul total égal : le maître, deux réseaux de la recette des réseaux de K4 (−0,18 pris seuls à la clé
47), devrait gagner sur eux un peu moins que K2 sur les siens (+1,42 ; ses réseaux, moins entraînés, sont un peu moins
divers, voir la partie B), soit de +1,0 à +1,2 sur le réseau seul : deux réseaux qui se partagent la moitié du calcul
battent le réseau seul nourri de tout (**S7**, 90 %). L'élève reçoit l'autre moitié ; si la moitié du calcul suffit au
réseau seul (**S9** : SEUL_H vaut SEUL10 à 0,5 point près, 30 % — établir une équivalence demande plus que de ne pas
voir d'écart : pour un réseau pris seul, l'écart-type par graine face au réseau seul était d'environ un point à la clé
47, et l'équivalence à 0,5 point près n'a alors qu'une chance sur trois d'être établie, même sans écart vrai), il part
du niveau du réseau seul et gagne ce que son maître lui donne : à la clé 47, l'élève avait pris 1,01 point des 1,84 de
son maître, soit un peu plus de la moitié ; ici j'attends de +0,3 à +0,9 sur son jumeau sans maître (**S6**, 85 %), d'où
**P2** (PHOTO2 bat le réseau seul, 60 % — moins sûr : un maître plus faible, un élève à moitié de calcul) ; le maître
bat sa photographie (**S8**, 80 %).

*Partie B.* Couper en deux a un prix — chaque réseau reçoit la moitié du calcul — et un gain — la moyenne de deux
réseaux. Le prix grandit quand le calcul manque : d'après les réseaux de la clé 47 pris seuls, diviser par deux coûte
0,20 point de 1/2 à 1/4 du calcul, 0,25 de 1/4 à 1/8, puis 1,51 en deux divisions de 1/8 à 1/32. Le gain diminue, parce
que des réseaux moins entraînés sont moins divers : à la clé 47, un ensemble gagnait sur ses propres réseaux, en points,
de 47 à 63 fois son ambiguïté (en unités de Brier), et la variance d'un réseau — autour de la moyenne de tous les
réseaux possibles de sa recette : l'ambiguïté divisée par (k − 1)/k — tombait de 0,061 pour les réseaux nourris de la
moitié du calcul à 0,048 (1/4), 0,038 (1/8) et 0,023 (1/32). Deux réseaux nourris chacun du huitième (K2_Q) devraient
gagner de 0,9 à 1,1 point sur eux-mêmes, pour un prix d'environ 0,25 : **P3** (K2_Q bat SEUL_Q, de +0,3 à +1,2, 85 %) ;
quatre réseaux nourris chacun du seizième (K4_Q), un gain d'environ 1,3 contre deux divisions d'environ un point en
tout : **S10** (55 %). Au seizième, deux réseaux nourris chacun du trente-deuxième (la recette des réseaux de K32) ne
devraient plus gagner que de 0,5 à 0,7 point sur eux-mêmes, pour un prix de 0,7 à 0,9 : le réseau seul devrait tenir, ou
l'emporter — **P4** (K2_S bat SEUL_S) n'a que 25 % ; quatre réseaux nourris chacun du soixante-quatrième : le prix (deux
divisions, environ deux points) dépasse de loin le gain (**S11**, le réseau seul bat K4_S, 85 %). D'où l'interaction :
ce que la coupe en deux gagne sur le réseau seul est plus grand à tout le calcul qu'au seizième (**S12**, 90 %), parce
que l'ensemble gagne moins sur ses réseaux (**S13**, 90 %), parce qu'ils sont moins divers (**S15**, 95 %), et parce que
le prix d'une division grandit (**S14**, 85 %). Ce sont mes explications, et chacune a sa mesure : si la mesure la
dément, l'explication tombe.

**Principales** (corrigées ensemble, voir la règle) :

| | Prédiction | Graines | Ampleur attendue | Confiance |
|---|---|---|---|---|
| **P1** | V_cal(MOY_TOILE4) > V_cal(SEUL10) — la moyenne des poids des quatre instantanés bat le réseau seul, au même calcul et au même coût d'usage | 124–147 | +0,2 à +0,9 | 60 % |
| **P2** | V_cal(PHOTO2) > V_cal(SEUL10) — la distillation à calcul total égal bat le réseau seul, au même coût d'usage | 124–147 | −0,1 à +0,9 | 60 % |
| **P3** | E_cal(K2_Q) > V_cal(SEUL_Q) — au quart du calcul, la coupe en deux bat le réseau seul | 100–123 | +0,3 à +1,2 | 85 % |
| **P4** | E_cal(K2_S) > V_cal(SEUL_S) — au seizième du calcul, la coupe en deux bat le réseau seul | 100–123 | −0,7 à +0,4 | 25 % |

**Secondaires** :

| | Prédiction | Graines | Ampleur attendue | Confiance |
|---|---|---|---|---|
| **S1** | V_cal(MOY_TOILE8) > V_cal(SEUL10) — la moyenne des poids des huit instantanés bat le réseau seul | 124–147 | −0,5 à +0,8 | 45 % |
| **S2** | \|V_cal(MOY_TOILE4) − E_cal(TOILE4)\| < 0,5 point — la moyenne des poids vaut l'ensemble des instantanés (équivalence) | 124–147 | −0,4 à +0,2 | 75 % |
| **S3** | E_cal(TOILE4) > V_cal(SEUL10) — l'ensemble des quatre instantanés bat le réseau seul | 124–147 | +0,3 à +1,2 | 90 % |
| **S4** | V_fin(SEUL10) > V_fin(MOY_TOILE4) — au seuil par défaut, la moyenne des poids perd | 124–147 | +1,5 à +3,5 | 97 % |
| **S5** | LOG(SEUL10) > LOG(MOY_TOILE4) — les logits de la moyenne des poids sont plus petits | 124–147 | de 5 à 30 % plus petits (le test porte sur l'écart en unités de logit) | 70 % |
| **S6** | V_cal(PHOTO2) > V_cal(SEUL_H) — l'élève bat son jumeau sans maître (mêmes poids de départ, mêmes paquets, même calcul) | 124–147 | +0,3 à +0,9 | 85 % |
| **S7** | E_cal(K2_H) > V_cal(SEUL10) — deux réseaux qui se partagent la moitié du calcul battent le réseau seul nourri de tout | 124–147 | +0,6 à +1,6 | 90 % |
| **S8** | E_cal(K2_H) > V_cal(PHOTO2) — le maître bat sa photographie | 124–147 | +0,2 à +1,2 | 80 % |
| **S9** | \|V_cal(SEUL_H) − V_cal(SEUL10)\| < 0,5 point — la moitié du calcul vaut tout le calcul, pour le réseau seul (équivalence) | 124–147 | −0,4 à +0,3 | 30 % |
| **S10** | E_cal(K4_Q) > V_cal(SEUL_Q) — au quart du calcul, la coupe en quatre bat le réseau seul | 100–123 | −0,2 à +0,9 | 55 % |
| **S11** | V_cal(SEUL_S) > E_cal(K4_S) — au seizième du calcul, le réseau seul bat la coupe en quatre | 100–123 | +0,5 à +2,5 | 85 % |
| **S12** | GAIN.K2_47 > GAIN.K2_S — ce que la coupe en deux gagne sur le réseau seul est plus grand à tout le calcul qu'au seizième | 100–123 | +0,8 à +2,4 | 90 % |
| **S13** | GMEM.K2_47 > GMEM.K2_S — ce que l'ensemble de deux réseaux gagne sur ses réseaux pris seuls est plus grand à tout le calcul qu'au seizième | 100–123 | +0,4 à +1,2 | 90 % |
| **S14** | PRIX.S > PRIX.Q — diviser par deux le calcul d'un réseau lui coûte plus au seizième qu'au quart | 100–123 | +0,2 à +1,0 | 85 % |
| **S15** | AMB(K2, clé 47) > AMB(K2_S) — deux réseaux nourris de la moitié du calcul sont plus divers que deux réseaux nourris du trente-deuxième | 100–123 | +0,012 à +0,025 | 95 % |

Toutes sur la vérité à seuil égal (réglé sur l'ancre), en points (centièmes), sauf S4 (au biais 0), S5 (en unités de
logit) et S15 (en unités de Brier).

**Trois lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « à calcul égal et au même coût d'usage, un seul réseau recueille une part du multiple »** : au moins une de
  P1, P2, S1 confirmée. Je prédis que oui (80 %).
- **L2 — « le découpage gagne encore quand le calcul manque »** : P4 confirmée. Je prédis que non (75 %).
- **L3 — « la moitié du calcul suffisait au réseau seul, et le découpage gagne moins quand le calcul manque »** : S9 et
  S12 confirmées. Je prédis que non (75 %) : S12 devrait tenir, mais S9 demande d'établir une équivalence que 24 graines
  permettent à peine. L3 serait compatible avec la lecture « saturation » sans l'établir : S12 est aussi prédite par la
  diversité (S13, S15).

**Ce que chaque issue voudra dire.**

- Si P1 tient : l'écart lu sans prédiction à la clé 47 se refait, prédit, sur des graines neuves — la moyenne des poids
  sur un taux cyclique est une recette : un seul réseau, au même calcul et au même coût d'usage, recueille une part du
  multiple. S'il est indécidable : le +0,70 de la clé 47 devait une part au choix après coup ; l'intervalle dira ce
  qu'il en reste. S'il est démenti : il était un hasard.
- Si P2 tient : à calcul total égal, la distillation fait entrer le multiple dans un seul réseau. Si S6 tient sans P2 :
  le maître aide l'élève, mais pas assez pour rendre ce que coûte la moitié du calcul donnée au maître.
- Si P3 tient et que P4 ne tient pas : le découpage gagne tant que le réseau seul a du calcul à revendre, pas quand le
  calcul manque. Si P4 tient : même au seizième du calcul, deux réseaux battent un réseau — le gain de la moyenne couvre
  encore le prix d'une division.
- Si S12 est démentie ou indécidable avec P4 confirmée : ce que gagne le découpage ne dépend pas du calcul comme je le
  crois, et la lecture « saturation » de la clé 47 est à revoir. Si S13, S14 ou S15 est démentie : le mécanisme
  correspondant (le gain de la moyenne, le prix d'une division, la diversité) ne joue pas comme je l'écris.
- Si S9 est démentie par le bas (la moitié du calcul coûte plus de 0,5 point au réseau seul) : le +0,02 de la clé 47 —
  la moyenne de deux réseaux — était une chance, et le réseau seul n'avait pas de calcul de trop. Par le haut (la moitié
  du calcul fait mieux de plus de 0,5 point) : la recette du réseau seul nourri de tout le calcul le dessert (son taux
  ou sa durée), et sa « saturation » serait un mauvais réglage.
- Si S5 est démentie : les logits de la moyenne ne sont pas plus petits que ceux du réseau seul ; l'analyse descriptive
  dira si la moyenne ne rétrécit pas les logits de ses instantanés — le raisonnement par les normes tomberait — ou si
  les instantanés n'ont pas l'échelle du réseau seul (seul le dernier instantané a son échelle mesurée : un repère, pas
  les quatre).

## La règle de décision

Toutes les comparaisons sont appariées par graine, sur les 24 graines de leur partie. Pour chaque test, l'écart par
graine d = A − B, le t de Student et la valeur p unilatérale exacte à n − 1 = 23 degrés de liberté.

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre. *Confirmée*
  si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ; *indécidable* sinon.
- **Tests secondaires (S1, S3–S8, S10–S15)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ;
  *indécidable* sinon.
- **Tests d'équivalence (S2, S9)**, marge ±0,5 point, deux tests unilatéraux (TOST) : *confirmée* si l'écart est établi
  au-dessus de −0,5 et au-dessous de +0,5 (les deux p < 0,025) ; *démentie* si l'écart est établi hors de la marge (p <
  0,025 pour « écart > +0,5 » ou pour « écart < −0,5 ») ; *indécidable* sinon.
- Les p ajustés par Holm sur les tests complets (19 si rien ne manque) sont rapportés pour information.
- Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
- Une graine dont l'écart n'est pas un nombre fini compte comme manquante. Plus de deux graines manquantes pour un
  test : ce test est *incomplet* ; un test principal incomplet compte p = 1 dans la correction de Holm, qui porte
  toujours sur les quatre.
- Une lecture « au moins une » (L1) tient si l'un de ses tests est confirmé, ne tient pas si aucun ne l'est et qu'aucun
  n'est incomplet, et reste incomplète sinon ; une lecture « toutes » (L2, L3) ne tient pas si l'un de ses tests
  complets n'est pas confirmé, tient si tous sont confirmés, et reste incomplète sinon.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %, et aucune
  équivalence ne sera lue hors de S2 et S9.
- Le plan doit tourner tel qu'écrit : une course absente, ou en double avec des valeurs différentes, sera signalée ; les
  courses de référence de la clé 47 doivent avoir les empreintes inscrites dans `verdict48.py` ; toutes les courses de
  la clé 48 d'une même graine, l'élève de PHOTO2 compris, doivent avoir la même empreinte du monde (l'état de son
  générateur, la réserve déjà tirée, le tampon et les ancres, à la fin de l'entraînement) : un écart sera rapporté.
  Entre les clés 47 et 48, le même monde tient au code (même moteur, mêmes graines), sans empreinte pour le vérifier.

Le script `verdict48.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle et calcule les trois
lectures. `ana48.py`, l'analyse descriptive et les comparaisons sans prédiction (chaque instantané des Toiles, les
moyennes de la seconde moitié, les biais réglés, la confiance, la courbe du calcul de la partie B et les recettes égales
d'une clé à l'autre, la décomposition des écarts en rappel du chat et autres classes, les durées), est écrit avant les
courses et figure dans le manifeste.

## Ce qui a tourné avant ce texte

- `essai48.py` (petit modèle du calcul : 2 pas par génération au lieu de 100, un juge de 2 500 images, 4 générations, 8
  pour TOILE8 ; graines 321 à 342, hors plan) : chacun des onze bras, coupé après une sauvegarde de son point de reprise
  puis repris, donne le même résultat, caractère par caractère, que la course faite d'une traite — à 31 points de coupe
  en tout (après la première génération, au milieu des mesures finales, à la dernière sauvegarde ; pour PHOTO2 aussi
  après les probabilités du maître et pendant la distillation ; pour TOILE8 au milieu des cycles). Les chemins communs
  refont les courses des clés 46 et 47, sur tout ce qu'elles ont en commun : course_coupe48 refait K2 et K4P (le maître
  et l'élève) de la clé 47, course_toile48 sa Toile ; le réseau seul de course_coupe48 refait course_seul de la clé 46 à
  1, 1/2, 1/4 et 1/16 du calcul (68 valeurs du trajet et du juge, aucun écart ; mesures finales et recyclages
  identiques) ; et de même sur 12 générations, la bascule ×30 de la 10e génération comprise (SEUL_S contre course_seul,
  K2 contre course_coupe, la Toile à quatre cycles de trois générations contre course_toile). L'élève de PHOTO2 et
  SEUL_H partent des mêmes poids, finissent avec leurs générateurs dans le même état (les mêmes paquets), dans le même
  monde (mêmes recyclages, mêmes ancres), et leurs trajets diffèrent. L'empreinte du monde est la même pour six bras
  d'une même graine et l'élève de PHOTO2, et une autre graine en a une autre — `essai48_bilan.json`.
- `essai48.py --vrai` (le vrai calcul, 2 générations, graines 321 à 325, et 398 et 399 pour les durées) : les mêmes
  identités (sauf K4P ; la Toile à deux cycles), aucun écart (34 valeurs du trajet et du juge pour chaque réseau seul) ;
  et les durées dans le bac à sable : 0,035 seconde par pas de gradient pour un réseau, 0,070 pour deux réseaux à la
  fois, 0,162 pour quatre (mesurées pendant qu'un autre calcul occupait le second cœur) ; une course de deux générations
  avec toutes ses mesures finales, 26 secondes pour SEUL_S et 78 pour K4_S : le coût fixe d'une course est petit —
  `essai48_bilan_vrai.json`. D'où, avec les durées des sessions cloud de la clé 47 (médianes : la Toile, un réseau
  nourri de tout le calcul, 2,1 heures ; K2, 3,0 ; K4P, 5,5), de 13 à 14 heures de calcul d'un cœur par tranche, de 310
  à 340 en tout, et de trois heures et demie à quatre heures par tranche sur quatre cœurs (jusqu'à six ou sept sur une
  machine lente).
- `controle48.py --ecrire` puis `controle48.py` : les valeurs attendues du contrôle d'environnement, sur la graine 200
  (hors plan), écrites dans un fichier sans être affichées, puis le contrôle lui-même (« bon » pour les quatre cas, puis
  « CONTRÔLE : bon », à 4 h 22).
- `synthese48.py` puis `verdict48.py` et `ana48.py` sur ses données de synthèse (chaque bras fait une fois au petit
  modèle, graine 401, et copié sur les 24 tranches avec des fronts bruités au hasard ; aucune donnée nouvelle), avec les
  courses réunies de la clé 47 : les scripts lisent toutes les mesures et appliquent la règle — les 264 courses lues,
  les 19 tests et les trois lectures appliqués ; l'empreinte du monde signale, comme elle le doit, les graines de la
  partie A, parce que la Toile à huit cycles des données de synthèse a couru huit générations et les autres bras
  quatre ; et les figures du journal se construisent sur ces données.
- Un essai à la main du code neuf, au petit modèle (graine 333, 4 générations) : SEUL_S, K4_S, PHOTO2 et TOILE4
  tournent. J'y ai vu des valeurs de la confiance de ces réseaux de quelques pas — par exemple, l'échelle des logits de
  la moyenne des quatre instantanés de TOILE4 plus petite que celle du dernier (0,83 contre 0,98) : un essai du code,
  pas une donnée (à ce calcul, les réseaux n'ont presque rien appris), mais je le dis, puisque S5 porte sur cette
  mesure.
- Une relecture indépendante du code et de ce texte, avant le dépôt, par un agent qui n'avait écrit ni l'un ni l'autre
  (de 2 h 50 à 3 h 20). Elle n'a trouvé aucun défaut qui change le calcul des courses. Elle a relevé quatre défauts du
  verdict, corrigés : le contrôle « même monde » ne contrôlait rien (la liste des recyclages est la même pour toutes les
  graines) — remplacé par une empreinte du monde ; un écart non fini aurait donné un verdict faux ; un test principal
  incomplet aurait changé la correction de Holm des trois autres ; les lectures se déclaraient incomplètes trop tôt. Et
  deux passages du raisonnement, corrigés : mon explication de la perte de la moyenne au seuil par défaut (des logits
  plus petits) était fausse en logique et démentie par les instantanés de la clé 47 ; la confiance de S9 était trop
  haute pour un test d'équivalence sur 24 graines. Elle a aussi demandé un cas du contrôle sur 12 générations, la
  reprise d'une tranche dans un conteneur recréé, et les consignes dans le manifeste. Une contre-vérification de mes
  corrections par le même agent (de 3 h 28 à 3 h 37) les a trouvées justes, et a relevé deux décalages qu'elles avaient
  introduits ou laissés, corrigés aussi : la consigne du contrôle ne montrait que trois « bon » sur quatre, et la règle
  écrite ici ne disait pas trois règles que le script du verdict applique désormais. Après ces corrections, les essais,
  le contrôle et le passage à blanc ci-dessus ont été refaits sur le code déposé.
- J'ai lu, pour la section « Ce que la clé 47 montre déjà », les courses publiées de la clé 47, son verdict, son analyse
  descriptive et ses comparaisons après coup, et j'y ai calculé, pour ce texte, les écarts de la moyenne des poids aux
  découpages en deux et en quatre, et les biais réglés et vérités au biais 0 des instantanés.
- Rien d'autre du plan.

## Où et comment

Les tranches (J1 à J24 ; onze courses chacune — la graine 123 + i de la partie A et la graine 99 + i de la partie B pour
la tranche Ji —, sur quatre processus) tournent dans des sessions cloud ouvertes sur le compte de Lucas, à sa demande,
sur ses crédits ; chacune exécute une tranche telle qu'écrite (`cle48/CLAUDE.md`), avec le lanceur à points de reprise
(`lancer48.py`), et pousse ses résultats sur sa branche du dépôt public de la clé 40
(https://github.com/Lucas2a63/cle40), où la clé 48 occupe le dossier `cle48/` ; le bac à sable ne fait que lire ce
dépôt. Les tranches peuvent partir en plusieurs fois, selon l'utilisation du compte ; le verdict attend les 24. Rien
n'agit hors de ce cadre : pas d'agent autonome, pas d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Un seul monde (fixe), une seule base d'images, une seule architecture, petite, en numpy, sans augmentation de données,
  en descente de gradient simple.
- **Les taux ne sont pas réglés** hors des deux points de la clé 40 (n = 1 et n = 32) : ils suivent la loi de puissance
  de la clé 47. Pour les réseaux de K4_S (1/64 du calcul, 6 000 pas), le taux (0,758 · LR0) est extrapolé au-delà du
  réseau court ; le balayage de la clé 40 avait trouvé, pour 12 000 pas, un plateau de 0,5 à 1 · LR0 (notes de l'ancre
  0,630 et 0,628). Le taux d'un cycle de la Toile, celui d'un réseau qui reçoit 1/4 (ou 1/8) du calcul, s'applique à un
  réseau déjà entraîné qui redémarre : il n'a pas été réglé non plus.
- Les réglages de la Photographie (température 4, une moitié de la perte pour le maître), le partage du calcul entre
  maître et élève (moitié-moitié), le nombre de cycles des Toiles (quatre et huit) et la « seconde moitié » des
  instantanés sont fixés d'avance, sans réglage.
- La partie B compare des courses de la clé 48 à des courses de la clé 47 (S12, S13, S15) : même moteur (les chemins
  communs refont la clé 47 au bit près, voir les essais), mêmes graines, même monde (garanti par le code : les courses
  de la clé 47 n'ont pas d'empreinte du monde) — mais des sessions et des machines différentes.
- P1 reprend un écart choisi après coup à la clé 47 : c'est son objet ; ses graines sont neuves. MOY_TOILE8 et les
  moyennes de la seconde moitié n'ont, elles, jamais été mesurées.
- La moyenne des poids n'est pas remise face à la recette coupée en deux ou en quatre à tout le calcul (à la clé 47,
  elles la devançaient de 0,74 et 1,14 point, sans test) : cette clé demande si un seul réseau recueille une part du
  multiple, pas s'il le recueille tout. Elle ne règle pas non plus les taux des découpages, et n'essaie pas de réseau
  plus large.
- La confiance (LOG) est une mesure nouvelle, sans donnée de la clé 47 : S5 est prédite sans repère.
- La marge des tests d'équivalence (0,5 point) est un choix, pas une grandeur tirée des données.
- 24 graines par partie : un écart d'un tiers de point serait à la limite de ce que la clé peut établir (écarts-types
  par graine de 0,5 à 1,0 point à la clé 47).
