# Clé 47 — À calcul égal, un réseau ou plusieurs ?

Pré-enregistrement, écrit le 23 octobre 2026 entre 7 h et 8 h 40 (heure de Paris), avant toute course du plan.
Projet Singularité (Lucas).

## La question

À la clé 46, sur le réseau convolutif et à seuil égal, le réseau seul nourri de tout le calcul d'entraînement (SEUL10)
a battu l'ensemble des 32 réseaux de la recherche aléatoire de 1,88 point ; choisir à la fin les réseaux de l'ensemble
l'a ramené à 0,24 point du réseau seul, sans écart établi. La clé n'a pas séparé ce qui fait perdre la population : le
partage du calcul (chaque réseau en reçoit 1/32), les réglages tirés au hasard, dans une plage qui n'est pas faite pour
le convolutif, le taux constant au lieu d'un cosinus, la curation. Cette clé pose la question directement, à calcul
d'entraînement égal :

1. **La meilleure recette convolutive, coupée en k réseaux nourris chacun de 1/k du calcul et lus en ensemble,
   fait-elle mieux que le réseau seul ?** k = 2, 4, 8, 32.
2. **Peut-on faire entrer le multiple dans un seul réseau, au même calcul et au même coût d'usage ?** Quatre façons
   connues de la littérature, que la lecture de *La Doctrine secrète* (23 octobre) a désignées par des images du
   livre : le **Veilleur** (la moyenne des poids le long de l'entraînement), la **Soupe** (la moyenne des poids
   d'enfants d'un même parent), la **Photographie** (un ensemble distillé dans un seul réseau — elle coûte, elle, le
   calcul du maître en plus), et la **Toile** (les instantanés d'un seul entraînement au taux cyclique — un ensemble,
   qui coûte plusieurs réseaux à l'usage).
3. **Une population aux réglages faits pour le convolutif**, sans curation, fait-elle mieux que celle de la clé 46 ?

## Ce que la clé 46 montre déjà (lu avant ce texte, sur ses courses publiées ; graines 100 à 123)

- À seuil égal (réglé sur l'ancre) : le réseau seul pesé à la 10e (SEUL10) 0,660 ; pesé dès le départ (SEUL) 0,654 ;
  le grand réseau 0,637 ; le réseau court, 32 fois moins de calcul (COURT, pesé dès le départ) 0,632 ; l'ensemble des
  32 réseaux de la recherche aléatoire (TOUS) 0,641 ; le choix glouton 0,658.
- Écarts appariés (moyenne, et écart-type par graine) : SEUL10 − COURT +2,82 (0,59) ; SEUL − COURT +2,17 (0,76) ;
  GLOUTON − SEUL10 −0,24 (1,03) ; TOUS − SEUL10 −1,88 (1,19).
- Dans la recherche aléatoire, un réseau convolutif valait d'autant moins que son taux était haut (−8,6 points du bas
  au haut de la plage) et que sa curation était forte (−3,1) — après coup.
- Le taux réglé par le balayage de la clé 40 : 0,0625 · LR0 pour le réseau seul (384 000 pas), 0,5 · LR0 pour le
  réseau court (12 000 pas), sur une grille en puissances de 2.

## Le montage

Tout se passe dans le monde des clés 40 à 46 : CIFAR-10 en 32×32, le chat fatal (3 % du monde, ×30 pour le juge), 500
images par génération tirées avec remise dans les 35 000 images du flux, un tampon de 5 000, l'ancre de 200 images de
la réserve dont 40 chats (réutilisées une fois la réserve épuisée), le juge sur les 10 000 images de test, 120
générations, monde fixe. L'architecture est le **réseau convolutif de la clé 40** (32×32×3 → 16×16×16 → 16×16×32 →
8×8×32 → 10, 25 600 paramètres environ). Le moteur (`echelle47.py`) est celui de la clé 46, inchangé, avec une section
nouvelle pour les bras de cette clé, qui ont leurs points de reprise intégrés.

**Les graines** : **100 à 123** — celles de la clé 46, dont les courses publiées servent de référence, graine par
graine (POPC face à la recherche aléatoire de la clé 46 ; le VEILLEUR doit refaire au bit près son réseau seul).

**Le calcul** : chaque bras reçoit le calcul d'entraînement de SEUL10 — 384 000 pas de gradient d'un petit réseau
convolutif sur 64 images —, sauf la Photographie, dont l'élève reçoit autant que son maître (K4), soit deux fois en
tout. Le taux d'un réseau qui reçoit 1/k du calcul suit la loi de puissance qui passe par les deux taux réglés de la
clé 40 : **0,0625 · k^0,6 · LR0** (k = 1 : 0,0625 ; 2 : 0,095 ; 4 : 0,144 ; 8 : 0,218 ; 32 : 0,5), en cosinus sur ses
propres pas. Il n'a pas été réglé pour k = 2, 4 et 8 (voir les limites).

**Les bras** (huit courses par graine ; 192 courses) :

| Bras | Ce que c'est |
|---|---|
| **K2, K4, K8, K32** | La recette de SEUL10 (taux en cosinus, décroissance WD0, sans curation, tirage uniforme dans le tampon, ×30 dans la perte à partir de la 10e génération) coupée en k réseaux indépendants — initialisations et paquets propres —, chacun nourri de 1/k du calcul (3 200/k pas par génération), au taux 0,0625 · k^0,6 · LR0 ; lus en ensemble (la moyenne de leurs probabilités, comme les 32 de la clé 46). Mesurés aussi : chacun de ses réseaux pris seul, et la moyenne de leurs poids (la soupe sans ancêtre commun). |
| **PHOTO** (avec K4, dans la même course, K4P) | Après K4 : les probabilités de l'ensemble K4, adoucies à la température 4, sur les 35 000 images du flux ; puis un seul réseau, à la recette de SEUL10 — même générateur, donc même initialisation et mêmes paquets que SEUL10, même monde, même ancre —, entraîné sur la perte w(y)·[½ entropie croisée + ½ · 4² · KL(maître ‖ élève) à la température 4], où w(y) est le poids de classe de SEUL10 (×30 au chat à partir de la 10e génération). |
| **POPC** | La recherche aléatoire convolutive de la clé 46 (32 réseaux, sans sélection, ×30 à partir de la 10e génération, taux constant), aux réglages tirés dans une plage faite pour le convolutif — taux de 0,005 à 0,05 (autour de 0,25 · LR0, le taux moyen d'un cosinus parti du taux réglé du réseau court), décroissance de 3,2·10⁻⁵ à 3,2·10⁻⁴, log-uniformes — et **sans curation** ; suivie du choix final de la clé 45 (TOUS, GLOUTON, …). |
| **VEILLEUR** | La course de SEUL10, refaite (elle doit être identique au bit près à celle de la clé 46), avec deux moyennes de ses poids tenues à côté sans rien changer à son calcul : **SWA**, la moyenne des poids de la fin de chacune des 30 dernières générations ; **EMA**, la moyenne mobile des poids, pas à pas (β = 0,9999). |
| **TOILE** | Un seul réseau, le calcul et le générateur de SEUL10 (même initialisation, mêmes paquets), mais un taux en cosinus qui redémarre quatre fois — quatre cycles de 30 générations, chacun de 0,144 · LR0 (le taux d'un réseau qui reçoit le quart du calcul) à 0 ; un instantané des poids à la fin de chaque cycle. Mesurés : l'ensemble des quatre instantanés, le dernier (le réseau final), chacun, la moyenne de leurs poids. |
| **SOUPE** | La course de SEUL10 jusqu'à mi-course, au bit près (le parent, à la fin de la 60e génération), puis quatre copies du parent, les enfants, entraînées chacune avec ses propres paquets sur le quart du calcul restant (800 pas par génération), au taux qui continue le cosinus de SEUL10 de sa valeur de mi-course à 0, sur leurs propres pas. Mesurés : **la soupe** (la moyenne des poids des quatre enfants), la soupe gourmande (Wortsman et coll., 2022 : les enfants rangés par leur critère seul sur l'ancre — celui du choix final de la clé 45 —, ajoutés un à un si le critère de la soupe ne baisse pas), l'ensemble des enfants, chacun. |

**Le seuil réglé sur l'ancre**, pour chaque ensemble et chaque réseau, est celui des clés 42 à 46 : la médiane basse
des biais du chat qui maximisent la note de l'ancre sur les ancres réunies des dix dernières générations ; la vérité
×30 du juge à ce biais, V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87.

**Les mesures** (par course) : **E_cal** (un ensemble) et **V_cal** (un réseau seul) : la vérité ×30 du juge au seuil
réglé sur l'ancre — la mesure principale ; E_fin / V_fin au biais 0 ; E_max / V_max au meilleur biais du juge (un
diagnostic) ; **AMB**, l'ambiguïté d'un ensemble sur le juge. SEUL10 est le réseau du VEILLEUR.

**Le coût d'usage** (multiplications par image à l'usage) : un réseau pour SEUL10, SWA, EMA, PHOTO, la Soupe ; k
réseaux pour Kk, quatre pour la Toile et l'ensemble des enfants, 32 pour POPC.

## Les prédictions

Mon raisonnement. Couper le calcul a un prix et un gain. Le prix : chaque réseau reçoit moins de pas ; d'après la clé
46, passer de 384 000 à 12 000 pas coûte de 2,2 à 2,8 points à un réseau seul, soit, si la perte est régulière en
logarithme du calcul, environ 0,4 à 0,55 point par division par deux. Le gain : la moyenne de réseaux indépendants
efface une part de leurs erreurs — sur le convolutif, à la clé 46, leur ambiguïté était forte (0,150). Pour deux à huit
réseaux, j'attends que les deux se compensent à peu près : un écart au réseau seul de quelques dixièmes de point, d'un
côté ou de l'autre, que la clé aura du mal à établir (P1, S1, S2, à 40–50 %) ; pour 32 réseaux, le prix l'emporte
(P2). Une population aux réglages faits pour le convolutif et sans curation devrait battre celle de la clé 46 (S4).
Faire entrer le multiple dans un seul réseau, à calcul égal : avec un taux en cosinus qui finit à 0, la fin de
l'entraînement fait déjà une moyenne des poids ; SWA et EMA ne devraient presque rien changer (S6, S7), ni la soupe
d'enfants nés à mi-course et moins entraînés que le réseau seul (P3). La soupe tient — ses enfants partent du même
point, alors que des réseaux nés de graines différentes n'ont rien en commun et que la moyenne de leurs poids
s'effondre (S10). La distillation, nourrie de deux fois le calcul, devrait aider un peu le réseau seul (S8). Les
instantanés de la Toile, mieux entraînés que des réseaux indépendants du quart du calcul mais plus semblables entre eux
(S13), font un ensemble que je ne sais pas départager de K4 (P4).

**Principales** (corrigées ensemble, voir la règle) — graines 100 à 123 :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | E_cal(K4) > V_cal(SEUL10) — la recette coupée en quatre, lue en ensemble, bat le réseau seul | −0,4 à +0,8 | 45 % |
| **P2** | V_cal(SEUL10) > E_cal(K32) — le réseau seul bat la recette coupée en 32 | −0,5 à +1,5 | 60 % |
| **P3** | V_cal(SOUPE) > V_cal(SEUL10) — la soupe bat le réseau seul, au même calcul et au même coût d'usage | −0,6 à +0,4 | 30 % |
| **P4** | E_cal(TOILE) > E_cal(K4) — les quatre instantanés battent quatre réseaux indépendants, au même calcul et au même coût d'usage | −0,6 à +0,6 | 45 % |

**Secondaires** :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | E_cal(K2) > V_cal(SEUL10) — coupée en deux | −0,3 à +0,5 | 40 % |
| **S2** | E_cal(K8) > V_cal(SEUL10) — coupée en huit | −0,6 à +0,8 | 40 % |
| **S3** | E_cal(K4) > E_cal(K32) — quatre réseaux battent trente-deux | +0,2 à +1,5 | 70 % |
| **S4** | E_cal(POPC, TOUS) > E_cal(INDEPC, TOUS, clé 46) — la plage faite pour le convolutif, sans curation, sert l'ensemble de la recherche aléatoire | +0,5 à +2,0 | 80 % |
| **S5** | E_cal(K32) > E_cal(POPC, TOUS) — à 32 réseaux, la recette réglée bat les réglages tirés au hasard | −0,5 à +1,0 | 55 % |
| **S6** | V_cal(SWA) > V_cal(SEUL10) — le Veilleur, SWA | −0,2 à +0,3 | 40 % |
| **S7** | V_cal(EMA) > V_cal(SEUL10) — le Veilleur, EMA | −0,2 à +0,3 | 40 % |
| **S8** | V_cal(PHOTO) > V_cal(SEUL10) — la Photographie (deux fois le calcul) bat le réseau seul | +0,0 à +1,0 | 60 % |
| **S9** | E_cal(K4) > V_cal(PHOTO) — le maître bat sa photographie | −0,5 à +0,8 | 45 % |
| **S10** | V_cal(SOUPE) > V_cal(MOY_K4) — la soupe d'un même parent bat la moyenne des poids de quatre réseaux sans ancêtre commun | +10 à +40 | 97 % |
| **S11** | E_cal(SOUPE.ENFANTS) > V_cal(SOUPE) — l'ensemble des enfants bat leur soupe | 0 à +0,8 | 55 % |
| **S12** | E_cal(TOILE) > V_cal(TOILE.DERNIER) — les quatre instantanés battent le dernier | +0,2 à +1,0 | 75 % |
| **S13** | AMB(K4) > AMB(TOILE) — quatre réseaux indépendants sont plus divers que quatre instantanés | +0,005 à +0,05 | 85 % |

Toutes sur la vérité à seuil égal (réglé sur l'ancre), sauf S13 (en unités de Brier). Les écarts de vérité sont en
points (centièmes).

**Deux lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « à calcul égal, la meilleure recette coupée en plusieurs réseaux, lus en ensemble, bat le réseau seul »** :
  au moins une de P1, S1, S2 confirmée. Je prédis que non (55 %).
- **L2 — « à calcul égal et au même coût d'usage, un seul réseau recueille le multiple »** : au moins une de P3, S6,
  S7 confirmée. Je prédis que non (70 %).

**Ce que chaque issue voudra dire.**

- Si L1 tient : à calcul égal, sur ce réseau et ce monde, plusieurs réseaux font mieux qu'un — la perte de la
  population à la clé 46 venait de ses réglages (taux, curation), pas du partage du calcul ; le coût d'usage reste à
  payer (k réseaux), et la Photographie dira s'il peut être rendu.
- Si L1 ne tient pas et que P2 tient : sur ce réseau, à calcul égal, le partage du calcul coûte au moins ce que la
  moyenne rapporte ; un réseau seul bien nourri reste le meilleur emploi du calcul.
- Si P1, S1 ou S2 est démentie : le réseau seul bat même les petits découpages — la moyenne de réseaux de la même
  recette rapporte moins que ce que coûtent les pas perdus.
- Si L2 tient : le multiple entre dans un seul réseau sans rien coûter de plus — la soupe, SWA ou EMA. Si L2 ne tient
  pas : à calcul égal, la fin d'un entraînement en cosinus fait déjà ce que ces moyennes feraient.
- Si S10 est démentie (la moyenne de réseaux sans ancêtre commun ne s'effondre pas) : ce petit réseau à tanh ne se
  comporte pas comme les réseaux de la littérature, et toute lecture des soupes est à revoir.

## La règle de décision

Toutes les comparaisons sont appariées par graine, sur les 24 graines. Pour chaque test, l'écart par graine
d = A − B, le t de Student et la valeur p unilatérale exacte à n − 1 = 23 degrés de liberté.

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ; *indécidable*
  sinon.
- **Tests secondaires (S1–S13)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ; *indécidable*
  sinon. Les p ajustés par Holm sur les 17 tests sont rapportés pour information.
- Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
- Plus de deux graines manquantes pour un test : ce test est *incomplet*.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %, et aucune
  équivalence ne sera lue sans test d'équivalence.
- Le plan doit tourner tel qu'écrit : une course absente, ou en double avec des valeurs différentes, sera signalée ;
  les courses de référence de la clé 46 doivent avoir les empreintes inscrites dans `verdict47.py`. Le réseau du
  VEILLEUR est comparé, graine par graine, au réseau seul SEUL10 de la clé 46 (trajet et mesures finales, au bit près),
  et le parent de la SOUPE à sa première moitié : un écart sera rapporté, et les tests garderont le réseau du VEILLEUR.

Le script `verdict47.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle et calcule les deux
lectures. `ana47.py`, l'analyse descriptive et les comparaisons sans prédiction (la courbe du découpage, les réseaux
pris seuls, les règles du choix final de POPC face à celles de la clé 46, les soupes sans ancêtre, la décomposition
des écarts en rappel du chat et autres classes, les durées), est écrit avant les courses et figure dans le manifeste.

## Ce qui a tourné avant ce texte

- `essai47.py` (petit modèle du calcul : 2 pas par génération au lieu de 100, un juge de 2 500 images, 4 générations ;
  graines 301 à 310, hors plan) : chacun des huit bras, coupé après une sauvegarde de son point de reprise puis repris,
  donne le même résultat, caractère par caractère, que la course faite d'une traite — à 18 points de coupe en tout (après
  la première génération, au milieu des mesures finales ; pour K4P aussi pendant la distillation et après la dernière
  sauvegarde ; pour la SOUPE aussi juste après la naissance des enfants). K1 (la recette coupée en un réseau) et le
  réseau du VEILLEUR refont au bit près la course SEUL10 du moteur de la clé 46 (68 valeurs du trajet, et toutes les
  mesures finales), et le parent de la SOUPE sa première moitié (34 valeurs) — `essai47_bilan.json`.
- `essai47.py --vrai` (le vrai calcul, 2 générations, graine 310) : les mêmes identités (34, 34 et 17 valeurs, aucun
  écart ; mesures finales identiques) ; et les durées d'une génération d'entraînement dans le bac à sable (deux
  processus sur deux cœurs) : 84 s pour un réseau seul, 89, 88, 90 et 106 s pour K2, K4, K8 et K32, 100 s pour POPC —
  `essai47_bilan_vrai.json`. D'où, avec les durées des sessions cloud de la clé 46, environ 22 heures de calcul d'un
  cœur par graine, 530 en tout, et de six à sept heures par tranche sur quatre cœurs.
- `controle47.py --ecrire` puis `controle47.py` : les valeurs attendues du contrôle d'environnement, sur la graine 200
  (hors plan), écrites dans un fichier sans être affichées, puis le contrôle lui-même (« bon » pour les trois cas,
  puis « CONTRÔLE : bon »).
- `synthese47.py` puis `verdict47.py` et `ana47.py` sur ses données de synthèse (chaque bras fait une fois au petit
  modèle, graine 401, et copié sur les 24 graines avec des fronts bruités au hasard ; aucune donnée nouvelle) : les
  scripts lisent toutes les mesures et appliquent la règle.
- J'ai lu, pour la section « Ce que la clé 46 montre déjà », les courses publiées de la clé 46 (graines 100 à 123) :
  les niveaux de ses bras et les écarts-types par graine de quatre écarts.
- Rien d'autre du plan.

## Où et comment

Les tranches (J1 à J24 ; une graine et huit courses chacune, sur quatre processus) tournent dans des sessions cloud
ouvertes sur le compte de Lucas, à sa demande, sur ses crédits ; chacune exécute une tranche telle qu'écrite
(`cle47/CLAUDE.md`), avec le lanceur à points de reprise dès le départ (`lancer47.py` ; leçon de la clé 46), et pousse
ses résultats sur sa branche du dépôt public de la clé 40 (https://github.com/Lucas2a63/cle40), où la clé 47 occupe le
dossier `cle47/` ; le bac à sable ne fait que lire ce dépôt. Rien n'agit hors de ce cadre : pas d'agent autonome, pas
d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Un seul monde (fixe), une seule base d'images, une seule architecture, petite, en numpy, sans augmentation de
  données, en descente de gradient simple.
- **Les taux des découpages ne sont pas réglés** : pour k = 2, 4 et 8, et pour la Toile, le taux vient d'une loi de
  puissance tirée des deux taux réglés de la clé 40 (k = 1 et k = 32), eux-mêmes choisis sur une grille en puissances
  de 2 et sur une graine ; un découpage mal réglé serait désavantagé. Les décroissances ne sont pas réglées non plus.
- La recette de 32 réseaux (K32) a la bascule à la 10e génération, alors que le taux du réseau court de la clé 40 a été
  réglé pesé dès le départ.
- Les réglages de la Photographie (température 4, une moitié de la perte pour le maître), de SWA (30 générations), de
  l'EMA (β = 0,9999), de la Soupe (quatre enfants, nés à mi-course) et de la Toile (quatre cycles) sont fixés d'avance,
  sans réglage. La Photographie coûte deux fois le calcul : S8 ne dit rien de l'avantage à calcul égal.
- POPC change deux choses à la fois par rapport à la recherche aléatoire de la clé 46 (la plage des réglages et la
  curation) : S4 ne les sépare pas.
- 24 graines : un écart d'un quart à un tiers de point serait à la limite de ce que la clé peut établir (écarts-types
  par graine de 0,6 à 1,2 point à la clé 46).
