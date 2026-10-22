# Clé 46 — L'ensemble sur l'architecture forte

Pré-enregistrement, écrit le 22 octobre 2026 entre 2 h et 3 h (heure de Paris), avant toute course du plan.
Projet Singularité (Lucas).

## La question

Depuis la clé 41, toutes les clés ont porté sur le seul perceptron. Sur lui, à seuil égal, l'ensemble des 32 réseaux
de la recherche aléatoire bat chacune des recettes standard remises à l'épreuve, de 1,9 à 2,2 points (clé 43) ; aucune
sélection essayée ne l'a établi plus haut (clés 43 à 45). Mais à la clé 40, sur le même monde et au seuil par défaut,
un seul petit réseau convolutif finissait de six à sept points au-dessus de cet ensemble de perceptrons, et l'ensemble
convolutif de la recherche aléatoire n'avait pas d'avance sur le meilleur réseau convolutif seul. Cette clé met les deux
à seuil égal, sur le réseau convolutif de la clé 40 :

1. **Sur l'architecture forte, l'ensemble des 32 réseaux de la recherche aléatoire — et le choix final de la clé 45 —
   gardent-ils, à seuil égal, une avance sur le réseau seul ?**
2. **L'architecture compte-t-elle plus que la population** — un seul réseau convolutif bat-il l'ensemble des 32
   perceptrons, et l'avance de l'ensemble est-elle plus petite sur le convolutif ?

## Ce que les clés 40, 43 et 45 montrent déjà (lu avant ce texte, sur les courses publiées)

- **Clé 40, convolutif, 12 graines (0 à 11), au seuil par défaut** (le seuil réglé sur l'ancre n'existait pas encore) :
  l'ensemble des 32 réseaux de la recherche aléatoire 0,641 ; le réseau seul pesé à la 10e génération 0,641, le
  réseau seul 0,637, le réseau court 0,630, le grand réseau 0,625 ; le réseau livré par la recherche aléatoire 0,599.
  Écarts appariés de l'ensemble : −0,01 [−0,52 ; +0,51] contre le réseau seul pesé à la 10e, +0,43 [−0,07 ; +0,94]
  contre le réseau seul, +1,08 [+0,54 ; +1,62] contre le réseau court, +1,63 [+1,16 ; +2,11] contre le grand réseau.
- **Clé 40, perceptron, 24 graines, au seuil par défaut** : l'ensemble de la recherche aléatoire 0,569, de 3,9 à 5,3
  points devant le système et les trois recettes de la clé 40.
- **Clés 43 et 45, perceptron, graines 100 à 123** (celles de cette clé), à seuil égal : l'ensemble des 32 (TOUS)
  0,571 ; le choix glouton 0,572 ; le réseau livré 0,539 ; le réseau court réglé à la clé 41 (COURT2) 0,551 ; le grand
  réseau 0,551 ; le réseau seul 0,540. Écart apparié de l'ensemble sur le réseau court réglé : +1,92 [+1,67 ; +2,18].
- Les réglages de la recherche aléatoire sont tirés au hasard une fois pour toutes (taux de 0,02 à 0,2, décroissance
  de 10⁻⁵ à 10⁻³, force de curation) ; ceux des recettes convolutives sont ceux du balayage de la clé 40 (le taux de
  départ seul, la décroissance à sa valeur centrale).

## Le montage

Tout se passe dans le monde des clés 40 à 45 : CIFAR-10 en 32×32, le chat fatal (3 % du monde, ×30 pour le juge), 500
images par génération tirées avec remise dans les 35 000 images du flux, un tampon de 5 000, l'ancre de 200 images de
la réserve dont 40 chats (réutilisées une fois la réserve épuisée), le juge sur les 10 000 images de test, 120
générations, monde fixe. L'architecture est le **réseau convolutif de la clé 40** : 32×32×3 → 16×16×16 → 16×16×32 →
8×8×32 → 10 (25 600 paramètres environ) ; le grand réseau a 96 et 186 filtres (283 000 paramètres), 32 fois plus
coûteux par image. Le moteur (`echelle46.py`) est celui de la clé 45, inchangé hors son en-tête.

**Les graines** : **100 à 123** (24 graines) — les premières des clés 43 à 45, dont les courses publiées du perceptron
servent de référence, graine par graine.

**Les bras** (tous refaits ici, sur chaque graine ; 120 courses) :

| Bras | Ce que c'est |
|---|---|
| **INDEPC** | La recherche aléatoire convolutive (32 réseaux, réglages tirés au hasard, avec curation, sans sélection ; ×30 dans la perte à partir de la 10e génération, la bascule du convolutif de la clé 40), suivie du choix final de la clé 45 : sept règles appliquées à ses 32 réseaux finaux — TOUS (les 32 à poids égaux), GLOUTON (la sélection gloutonne avec remise de Caruana), ELAGAGE, MEILLEURS8, MEILLEURS16, TETE, HASARD —, chacune avec son front du juge, son front de l'ancre et sa diversité ; et son réseau livré (LIVRE, le mieux noté par l'ancre de la dernière génération). |
| **SEUL10** | Le réseau convolutif seul, qui reçoit le calcul d'entraînement de toute la population (32 fois celui d'un de ses réseaux), taux en cosinus, pesé à ×30 à partir de la 10e génération : la recette standard munie de la bascule, la meilleure recette convolutive de la clé 40. |
| **SEUL** | Le même, pesé dès le départ. |
| **GRAND** | Le grand réseau convolutif, même calcul d'entraînement, même coût d'usage que l'ensemble des 32. |
| **COURT** | Le réseau convolutif seul, 32 fois moins de calcul d'entraînement. |

Les taux de départ des quatre recettes sont ceux du balayage de la clé 40 (`balayage40_choix.json` : 0,0625, 0,0625,
0,5 et 0,5 fois le taux central), la décroissance des poids à sa valeur centrale (10⁻⁴), comme à la clé 40.

**Le seuil réglé sur l'ancre**, pour chaque ensemble et chaque réseau, est celui des clés 42 à 45 : la médiane basse
des biais du chat qui maximisent la note de l'ancre sur les ancres réunies des dix dernières générations ; la vérité
×30 du juge à ce biais, V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87.

**Les mesures** (par course) :
- **E_cal** (une règle du choix final, dont TOUS) et **V_cal** (un réseau seul) : la vérité ×30 du juge au seuil
  réglé sur l'ancre — la mesure principale ; **E_fin** / **V_fin** au biais 0 ; **E_max** / **V_max** au meilleur
  biais du juge (un diagnostic).
- **AMB** : l'ambiguïté de l'ensemble des 32 sur le juge — l'écart quadratique moyen des probabilités des réseaux à
  leur moyenne —, ce que l'ensemble gagne sur le Brier moyen de ses réseaux.
- Pour le perceptron, les mêmes mesures sont lues dans les courses publiées de la clé 45 (TOUS, son choix final) et de
  la clé 43 (COURT2), sur les mêmes graines ; `verdict46.py` vérifie leurs empreintes.

**Le calcul** : 384 000 pas de gradient d'un petit réseau convolutif sur 64 images pour la population (32 réseaux ×
12 000 pas) comme pour le réseau seul (SEUL10, SEUL) ; 12 000 pas du grand réseau, 32 fois plus coûteux chacun — le
même calcul ; 12 000 pas pour le réseau court, 32 fois moins. 120 courses ; environ 200 heures de calcul d'un cœur, d'après les durées de la clé 40 (4,0 heures par course de la
recherche aléatoire convolutive, 1,6 par réseau seul, 0,9 pour le grand réseau, quelques minutes pour le réseau
court) et un essai de durée fait ici (98 secondes par génération d'entraînement de la population, 44 par passage du
juge ; 48 secondes par génération du réseau seul).

## Les prédictions

Mon raisonnement. Un ensemble compense surtout la faiblesse et la dispersion de ses membres. Un perceptron, qui voit
l'image comme une liste de pixels, apprend des traits fragiles, et ses erreurs varient d'un réseau à l'autre : la
moyenne en efface beaucoup — d'où, sur le perceptron, deux points d'avance à seuil égal sur toute recette. Un réseau
convolutif, qui partage ses filtres entre positions, apprend des traits plus stables ; ses erreurs se ressemblent
davantage d'un réseau à l'autre, et un seul réseau bien réglé capte déjà l'essentiel. La recherche aléatoire paie de
plus ses réglages tirés au hasard : ses réseaux sont en moyenne moins bons qu'une recette réglée. Sur le convolutif,
ce coût pourrait égaler ce que la moyenne rapporte — c'est ce que montrait la clé 40 au seuil par défaut. J'attends
donc pas d'avance établie de l'ensemble sur le réseau seul pesé à la 10e (P1, à 35 %), une avance nette sur le grand
réseau, au même coût d'usage (P2), un réseau convolutif seul loin devant l'ensemble des 32 perceptrons (P3), une avance
de l'ensemble plus petite sur le convolutif que sur le perceptron (P4), et que la moyenne gagne moins, en score de
Brier, sur ses réseaux convolutifs que sur ses perceptrons (S11) — si S11 est démentie, c'est cette explication par la
ressemblance des erreurs qui l'est.

**Principales** (corrigées ensemble, voir la règle) — graines 100 à 123 :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | E_cal(TOUS) > V_cal(SEUL10), convolutif — l'ensemble des 32 bat le réseau seul pesé à la 10e | −0,5 à +0,5 | 35 % |
| **P2** | E_cal(TOUS) > V_cal(GRAND), convolutif — l'ensemble des 32 bat le grand réseau, au même coût d'usage | +0,5 à +2,0 | 80 % |
| **P3** | V_cal(SEUL10, convolutif) > E_cal(TOUS, perceptron, clé 45) — un seul réseau convolutif bat l'ensemble des 32 perceptrons | +5 à +8 | 95 % |
| **P4** | [E_cal(TOUS) − V_cal(COURT2)]_perceptron − [E_cal(TOUS) − V_cal(SEUL10)]_convolutif > 0 — l'avance de l'ensemble sur le réseau seul est plus petite sur le convolutif | +1,5 à +2,5 | 85 % |

**Secondaires** :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | E_cal(TOUS) > V_cal(SEUL), convolutif — l'ensemble bat le réseau seul pesé dès le départ | −0,3 à +0,8 | 45 % |
| **S2** | E_cal(TOUS) > V_cal(COURT), convolutif — l'ensemble bat le réseau court | +0,3 à +1,5 | 70 % |
| **S3** | E_cal(GLOUTON) > E_cal(TOUS), convolutif — le choix glouton bat les 32 | −0,1 à +0,3 | 35 % |
| **S4** | E_cal(GLOUTON) > E_cal(HASARD), convolutif — le choix glouton bat autant de réseaux tirés au hasard | +0,1 à +0,6 | 65 % |
| **S5** | E_cal(TOUS) > E_cal(MEILLEURS8), convolutif — les 32 battent les 8 meilleurs pris un à un | +0,1 à +0,6 | 60 % |
| **S6** | E_cal(TOUS, convolutif) > E_cal(TOUS, perceptron) — l'ensemble convolutif bat l'ensemble des perceptrons | +5 à +8 | 97 % |
| **S7** | V_cal(SEUL10) > V_cal(SEUL), convolutif — la bascule sert le réseau seul | −0,2 à +0,8 | 50 % |
| **S8** | E_cal(GLOUTON) > V_cal(SEUL10), convolutif — le choix glouton bat le réseau seul pesé à la 10e | −0,5 à +0,6 | 35 % |
| **S9** | E_cal(TOUS) > V_cal(LIVRE), convolutif — l'ensemble bat le réseau livré par la recherche aléatoire | +2 à +5 | 92 % |
| **S10** | V_cal(SEUL10) > V_cal(GRAND), convolutif — le réseau seul pesé à la 10e bat le grand réseau | +0,5 à +2,0 | 70 % |
| **S11** | AMB(TOUS, perceptron) > AMB(TOUS, convolutif) — la moyenne gagne moins sur ses réseaux convolutifs | +0,005 à +0,04 | 70 % |

Toutes sur la vérité à seuil égal (réglé sur l'ancre), sauf S11 (en unités de Brier). Les écarts de vérité sont en
points (centièmes).

**Deux lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « sur le convolutif aussi, l'ensemble des 32 garde une avance sur le réseau seul »** : P1 et S1 confirmées.
  Je prédis que non (70 %).
- **L2 — « l'architecture compte plus que la population »** : P3 et P4 confirmées. Je prédis que oui (80 %).

**Ce que chaque issue voudra dire.**

- Si P1 tient : l'avance de l'ensemble n'est pas propre au perceptron — à seuil égal, 32 réseaux convolutifs aux
  réglages tirés au hasard battent la meilleure recette convolutive de la clé 40 ; le choix final pourrait alors en
  réduire le coût d'usage.
- Si P1 ne tient pas et que P3 tient : sur l'architecture forte, l'ensemble n'apporte pas, de façon établie, plus
  qu'un réseau seul bien réglé — la capacité vient de l'architecture, et l'avance de l'ensemble de perceptrons tenait
  à la faiblesse de ses membres.
- Si P1 est démentie : le réseau seul fait mieux que l'ensemble — sur le convolutif, ce que coûtent les réglages tirés
  au hasard dépasse ce que rapporte la moyenne.
- Si S11 est démentie : la moyenne ne gagne pas moins sur les réseaux convolutifs ; l'explication par la ressemblance
  de leurs erreurs ne tient pas, et un écart entre les avances (P4) viendrait d'ailleurs — la qualité des réseaux,
  leurs réglages.

## La règle de décision

Toutes les comparaisons sont appariées par graine, sur les 24 graines. Pour chaque test, l'écart par graine
d = A − B (pour P4, la différence de deux écarts), le t de Student et la valeur p unilatérale exacte à n − 1 = 23
degrés de liberté.

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ; *indécidable*
  sinon.
- **Tests secondaires (S1–S11)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ; *indécidable*
  sinon. Les p ajustés par Holm sur les 15 tests sont rapportés pour information.
- Des écarts tous nuls : t = 0, p = 0,5 dans les deux sens.
- Plus de deux graines manquantes pour un test : ce test est *incomplet*.
- Un test indécidable n'établit pas l'absence d'écart : il sera rapporté avec son intervalle à 95 %, et aucune
  équivalence ne sera lue sans test d'équivalence.
- Le plan doit tourner tel qu'écrit : une course absente ou en double avec des valeurs différentes sera signalée ; les
  courses de référence du perceptron doivent avoir les empreintes inscrites dans `verdict46.py`.

Le script `verdict46.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle, vérifie les
empreintes des courses reprises des clés 43 et 45, et calcule les deux lectures. `ana46.py`, l'analyse descriptive et
les comparaisons sans prédiction, est écrit avant les courses et figure dans le manifeste.

## Ce qui a tourné avant ce texte

- `identite46.py` : le moteur refait au bit près, sur le convolutif, des courses publiées de la clé 40, graine 0 : le
  réseau court entier (1 050 valeurs, aucun écart ; 166 valeurs nouvelles, le front de l'ancre) et les trois premières
  générations de la recherche aléatoire (96 valeurs, aucun écart ; 18 valeurs nouvelles, la diversité aux passages du
  juge) — `identite46_court.json`, `identite46_indep3.json`. Les répliques entières du réseau seul pesé à la 10e et du
  grand réseau, graine 0, ont été lancées avant ce texte et finiront pendant la campagne : leur résultat sera rapporté.
- Un essai de durée hors plan (graine 201 ; trois générations de la recherche aléatoire et du réseau seul pesé à la
  10e) : seules les durées ont été affichées.
- `controle46.py --ecrire` puis `controle46.py` : les valeurs attendues du contrôle d'environnement, sur la graine 200
  (hors plan), écrites dans un fichier sans être affichées, puis le contrôle lui-même (« bon » trois fois).
- `verdict46.py` et `ana46.py` ont été essayés sur des données de synthèse construites à partir des courses publiées
  du perceptron des clés 43 et 45, rebaptisées convolutives — aucune donnée convolutive nouvelle. `ana46.py`, l'analyse
  descriptive, a été écrit avant les courses et figure dans le manifeste.
- J'ai lu, pour la section « Ce que les clés 40, 43 et 45 montrent déjà », les courses publiées de la clé 40 (convolutif,
  graines 0 à 11 ; perceptron, graines 0 à 23) et celles du perceptron des clés 43 et 45 sur les graines 100 à 123.
- Rien d'autre du plan.

## Où et comment

Les tranches (H1 à H12 ; deux graines et dix courses chacune, sur quatre processus) tournent dans des sessions cloud
que je lance depuis l'ordinateur de Lucas, à sa demande, sur ses crédits ; chacune exécute une tranche telle qu'écrite
(`cle46/CLAUDE.md`) et pousse ses résultats sur sa branche du dépôt public de la clé 40
(https://github.com/Lucas2a63/cle40), où la clé 46 occupe le dossier `cle46/` ; le bac à sable ne fait que lire ce
dépôt. Rien n'agit hors de ce cadre : pas d'agent autonome, pas d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Un seul monde (fixe), une seule base d'images, des réseaux en numpy, sans augmentation de données, en descente de
  gradient simple ; deux architectures seulement, petites, et un seul grand réseau.
- Les recettes convolutives n'ont été réglées que sur le taux de départ (clé 40), pas sur la décroissance, quand les
  recettes du perceptron l'ont été sur les deux (clé 41) : la comparaison des avances (P4) oppose une recette
  convolutive moins réglée à une recette du perceptron mieux réglée — ce qui joue plutôt contre P4.
- Le réseau seul pesé à la 10e (SEUL10) est désigné d'avance comme la meilleure recette convolutive à partir de la
  clé 40, où il l'était au seuil par défaut et après coup ; c'est P1 qui la met à l'épreuve à seuil égal.
- Les références du perceptron viennent des clés 43 et 45 : mêmes graines, même moteur, mais l'architecture ne change
  pas que le réseau — la bascule du convolutif est à la 10e génération, celle du perceptron dès le départ.
- 24 graines : un écart d'un tiers de point entre l'ensemble et le réseau seul serait à la limite de ce que la clé
  peut établir.
