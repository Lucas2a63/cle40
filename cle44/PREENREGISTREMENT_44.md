# Clé 44 — Une sélection qui serve l'ensemble

Pré-enregistrement, écrit le 21 octobre 2026 vers 14 h 35 (heure de Paris), avant toute course du plan.
Projet Singularité (Lucas).

## La question

À la clé 43, à seuil égal — chacun réglant son seuil du chat sur l'ancre, sans voir le juge —, l'ensemble des 32
réseaux de la recherche aléatoire, une population sans sélection, finissait à 0,572 et battait les trois recettes
standard remises à l'épreuve, de 2,2 à 3,2 points. La sélection du système faisait gagner son réseau livré (+0,93 point
sur celui de la recherche aléatoire, après coup), mais coûtait 1,65 point à son ensemble ; elle réduisait la population à une seule
lignée — sans que le lien entre les deux soit testé, ni la diversité des réseaux mesurée. Deux questions, que cette clé
prend ensemble :

1. **Une sélection peut-elle servir l'ensemble** — classer les réseaux selon ce qu'ils lui apportent, et non selon
   leur note propre — au point de faire mieux que pas de sélection du tout ? En copiant les poids (ENS), ou seulement
   les réglages, les poids restant indépendants (ENSR) ?
2. **La sélection individuelle ôte-t-elle sa diversité à la population ?** La diversité est mesurée ici pour la
   première fois.

## Ce que la clé 43 montre déjà (lu avant ce texte, sur les courses publiées, graines 100 à 147)

- À seuil égal : ensembles — recherche aléatoire 0,572, sans curation 0,567, système 0,555, 32 recettes réglées
  0,552 ; réseaux — grand réseau 0,550, réseau court réglé 0,550, système 0,548, réseau seul 0,540, réseau livré par la
  recherche aléatoire 0,539.
- Ce que l'ensemble gagne sur son réseau livré : recherche aléatoire +3,3 points, système +0,7, 32 recettes +0,2. Au
  seuil par défaut, les réseaux de la recherche aléatoire valent 0,480 en moyenne, ceux des 32 recettes 0,550.
- Lignées du système : 6,2 en moyenne à la génération 8, 1,2 à la génération 40, une seule aux générations 80 et 119.
  Au seuil par défaut, l'ensemble du système devance celui de la recherche aléatoire jusqu'à la génération 32, puis
  plafonne.
- À seuil égal, l'ensemble du système bat celui des 32 recettes de 0,30 point (S12 démentie) ; au seuil par défaut,
  c'est l'inverse (−0,53).
- Aucun ensemble ne se surestime (note de l'ancre moins vérité : de −0,3 à −0,1 point) ; les réseaux livrés, si.

## Le montage

Tout se passe sur le **perceptron** (3 072 → 128 → 10), en **32×32**, dans le monde des clés 40 à 43 : CIFAR-10, le
chat fatal (3 % du monde, ×30 pour le juge), 500 images par génération tirées avec remise dans les 35 000 images du
flux, un tampon de 5 000, l'ancre de 200 images de la réserve dont 40 chats (réutilisées une fois la réserve épuisée),
le juge sur les 10 000 images de test, 120 générations, monde fixe. Le moteur (`echelle44.py`) est celui de la
clé 43, plus la diversité, le critère « apport à l'ensemble » et deux bras ; pour les bras des clés 41 à 43, il refait
leurs courses au bit près.

**Les graines** sont celles de la clé 43 : **100 à 147**. À graine égale, SYS, INDEP, ENS et ENSR partent des mêmes
réseaux et des mêmes réglages et voient le même monde ; SYS, ENS et ENSR font aussi les mêmes tirages de sélection et
de mutation — seul change ce que la sélection copie, et selon quel classement.

**Les bras**

| Bras | Ce que c'est |
|---|---|
| **ENS** (nouveau) | Le système, dont la sélection classe les 32 réseaux selon leur **apport à l'ensemble** sur l'ancre de la génération : le quart qui apporte le moins est remplacé par des copies (poids, préférences de curation, réglages mutés) du quart qui apporte le plus. Le réseau livré reste le mieux noté par l'ancre, et il n'est jamais remplacé (dans le système, il ne peut pas l'être). |
| **ENSR** (nouveau) | La même sélection, qui ne copie que les **réglages** (taux, décroissance, force de curation, mutés) : les poids et les préférences de curation des 32 réseaux restent les leurs. |
| **INDEP** | La recherche aléatoire, avec curation (sans sélection) — refaite. |
| **SYS** | Le système entier des clés 36 à 43 (sélection sur la note propre) — refait. |
| **M32** | 32 réseaux courts réglés (taux LR0/8 en cosinus, décroissance 3e-2), sans sélection ni curation — refaits. |
| **GRAND** | Le grand réseau de la clé 40 : ses courses de la clé 43, sur les mêmes graines, sont reprises telles quelles (leurs empreintes sont vérifiées par `verdict44.py`). |

INDEP, SYS et M32 sont refaits pour mesurer leur diversité : **ils doivent redonner au bit près toutes les valeurs de
leurs courses publiées de la clé 43**.

**L'apport à l'ensemble** (fixé ici, calculé par `contributions` dans `echelle44.py`). À chaque génération, après
l'apprentissage, sur l'ancre de la génération (les 200 images et leurs poids, ceux de la note de sélection) : la
log-vraisemblance pondérée de la bonne classe sous la probabilité moyenne des 32 réseaux,
L = Σᵢ wᵢ log p̄(yᵢ) / Σᵢ wᵢ, moins la même sous la moyenne des 31 autres réseaux. Un réseau apporte d'autant plus
qu'il donne de la probabilité à la bonne classe là où les autres en donnent peu.

**La diversité** (mesurée sur le juge, à chaque passage et sur la population finale) :
- **AMB**, l'ambiguïté : la moyenne, sur les 10 000 images et les 32 réseaux, de Σ_c (p_k,c − p̄_c)², l'écart
  quadratique des probabilités d'un réseau à leur moyenne. Le Brier de l'ensemble est le Brier moyen des réseaux moins
  l'ambiguïté. Elle est lue au passage 119 du juge, avant la dernière sélection (sur la population finale, 8 des 32
  réseaux de SYS et d'ENS sont des copies fraîches).
- Rapportés sans prédiction : l'ambiguïté sur les seuls chats, le désaccord (la part des couples réseau-image dont la
  classe la plus probable diffère de celle de l'ensemble), les scores de Brier.

**Le seuil réglé sur l'ancre**, pour un réseau comme pour un ensemble, est celui de la clé 43 (la médiane basse des
biais qui maximisent la note de l'ancre sur les ancres réunies des dix dernières générations ; la vérité ×30 du juge
à ce biais, V = w·Rf + (1 − w)·Ao, w = 0,9 / 1,87). Comme à la clé 43, les mesures finales sont prises sur la
population après la dernière sélection (pour SYS et ENS, 8 des 32 réseaux sont alors des copies) ; l'écart à
l'ensemble d'avant la dernière sélection, au seuil par défaut, sera rapporté.

**Le calcul** : 384 000 pas de gradient d'un petit réseau sur 64 images pour chaque population ; l'apport à
l'ensemble ne coûte qu'un passage de l'ancre. 240 courses ; environ 100 heures de calcul d'un cœur.

**Les mesures** (par course) :
- **E_cal** : la vérité ×30 du juge pour l'ensemble, à la génération 119, au seuil réglé sur l'ancre.
- **V_cal** : la même pour le réseau livré (pour GRAND, le réseau lui-même).
- **H_ens** : la note de l'ancre de l'ensemble moins sa vérité, en moyenne sur les 11 passages 40, …, 119.
- **AMB** : l'ambiguïté de la population au passage 119 du juge. **LIG** : le nombre de lignées (ancêtres distincts des
  poids) au même passage.

Rapportés sans prédiction : E_fin, E_max, E_vie, V_fin, V_max ; la diversité et les lignées au fil des passages du
juge ; pour ENS et ENSR, les apports et la dispersion des réglages au fil des générations ; l'ensemble avant et après
la dernière sélection, et la diversité de la population finale ; l'ensemble à rappel du chat égal ; le lien entre diversité et gain de l'ensemble sur son
réseau livré ; les comparaisons avec le réseau court réglé, le réseau seul et la recherche aléatoire sans curation de
la clé 43 ; les durées.

## Les prédictions

Mon raisonnement. Une sélection qui classe les réseaux selon leur apport à l'ensemble devrait remplacer d'abord les
réseaux redondants ou nuisibles et garder les réseaux utiles et différents. Mais tant qu'elle copie des poids, chaque
remplacement crée un doublon : les lignées fusionnent, plus lentement que dans le système ; et l'apport d'un réseau,
mesuré sur les 200 images d'une ancre, est bruité. J'attends donc d'ENS un meilleur ensemble que celui du système,
plus de lignées et plus de diversité — mais pas qu'il batte la recherche aléatoire, qui garde toute la sienne. Ne
copier que les réglages garde les 32 lignées de poids, mais rapproche les réglages de ceux des réseaux utiles : cela
peut aider comme nuire. Le système, réduit à une lignée, devrait être bien moins divers que la recherche aléatoire, et
les 32 recettes identiques aussi, quoique moins.

**Principales** (corrigées ensemble, voir la règle) — graines 100 à 147 :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | E_cal(ENS) > E_cal(INDEP) — à seuil égal, la sélection pour l'ensemble fait un meilleur ensemble que pas de sélection | 0 à +0,5 | 35 % |
| **P2** | E_cal(ENS) > E_cal(SYS) — la sélection pour l'ensemble fait un meilleur ensemble que la sélection individuelle | +0,5 à +1,5 | 80 % |
| **P3** | E_cal(ENSR) > E_cal(INDEP) — la sélection des seuls réglages pour l'ensemble fait un meilleur ensemble que pas de sélection | 0 à +0,5 | 40 % |
| **P4** | AMB(INDEP) > AMB(SYS) — la sélection individuelle réduit la diversité de la population | nettement | 95 % |

**Secondaires** :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | E_cal(ENS) > V_cal(GRAND) — l'ensemble d'ENS bat le grand réseau, au même coût d'usage | +0,5 à +2,0 | 85 % |
| **S2** | E_cal(ENSR) > E_cal(SYS) — l'ensemble d'ENSR bat celui du système | +0,5 à +2,0 | 85 % |
| **S3** | AMB(ENS) > AMB(SYS) — la sélection pour l'ensemble garde plus de diversité que la sélection individuelle | nettement | 85 % |
| **S4** | AMB(INDEP) > AMB(ENS) — mais moins que pas de sélection | — | 70 % |
| **S5** | AMB(INDEP) > AMB(M32) — 32 recettes réglées identiques sont moins diverses que la recherche aléatoire | — | 80 % |
| **S6** | LIG(ENS) > LIG(SYS) — la sélection pour l'ensemble garde plus de lignées | plusieurs lignées | 90 % |
| **S7** | \|H_ens(ENS)\| < 1 point — l'ensemble d'ENS ne se surestime (ni ne se sous-estime) pas de plus d'un point | — | 75 % |
| **S8** | V_cal(SYS) > V_cal(ENS) — la sélection individuelle livre un meilleur réseau | +0,2 à +1,0 | 65 % |
| **S9** | E_cal(ENS) > V_cal(ENS) — l'ensemble d'ENS bat son propre réseau livré | +1 à +3 | 95 % |
| **S10** | E_cal(ENSR) > E_cal(ENS) — ne copier que les réglages fait un meilleur ensemble que copier aussi les poids | −0,5 à +0,5 | 55 % |
| **S11** | AMB(INDEP) > AMB(ENSR) — la sélection des réglages pour l'ensemble réduit un peu la diversité | — | 60 % |
| **S12** | E_cal(ENS) > E_cal(M32) — l'ensemble d'ENS bat celui de 32 recettes réglées | +0,5 à +2,0 | 85 % |
| **S13** | E_cal(ENSR) > V_cal(GRAND) — l'ensemble d'ENSR bat le grand réseau | +1,0 à +2,5 | 90 % |

Toutes sur la vérité à seuil égal (réglé sur l'ancre), sauf P4, S3 à S5 et S11 (diversité), S6 (lignées) et S7
(surestimation). Les écarts de vérité sont en points (centièmes) ; ceux d'ambiguïté, en unités de Brier.

**Deux lectures écrites d'avance** (sans test supplémentaire) :

- **L1 — « une sélection peut servir l'ensemble »** : P1 ou P3 confirmée. Je prédis que non (55 %).
- **L2 — « la sélection individuelle ôte sa diversité à l'ensemble ; une sélection pour l'ensemble lui en rend une
  part et fait un meilleur ensemble »** : P4, S3 et P2 confirmées. Je prédis que oui (65 %).

**Ce que chaque issue voudra dire.**

- Si L1 tient : une sélection ajoute à l'ensemble ce que la recherche aléatoire ne donne pas — la première fois, dans
  ce projet, que la sélection elle-même ajoute à la meilleure capacité mesurée à seuil égal (sur ce monde fixe, ce
  perceptron).
- Si P3 tient et pas P1 : c'est la copie des poids qui coûte à l'ensemble, pas la sélection en soi.
- Si P2 tient mais ni P1 ni P3 : classer pour l'ensemble limite le tort de la sélection sans le racheter ; la meilleure
  population reste celle qu'on ne sélectionne pas.
- Si P4 et S3 tiennent : la perte de l'ensemble sous la sélection va avec une perte de diversité, mesurée — un lien,
  pas une preuve de cause.
- Si P1 est démentie : même orientée vers l'ensemble, une sélection qui copie des poids lui coûte.

## La règle de décision

Toutes les comparaisons sont appariées par graine, sur les 48 graines. Pour chaque test, l'écart par graine
d = A − B (pour S7, la valeur A elle-même), le t de Student et la valeur p unilatérale exacte à n − 1 = 47 degrés de
liberté.

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
- **Identité** : les 144 courses INDEP, SYS et M32 refaites doivent redonner au bit près toutes les valeurs de leurs
  courses publiées de la clé 43 (hors durée et tâche). Une seule valeur différente rend la clé invalide : je le dirai,
  et aucun verdict ne sera présenté comme tel.

Le script `verdict44.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle, vérifie
l'identité et les empreintes des courses reprises de la clé 43, et calcule les deux lectures. Il calcule le poids du
chat à partir des proportions du monde, comme le moteur (à 6·10⁻¹⁷ près de 0,9 / 1,87, ce qui ne change aucun
résultat).

## Ce qui a tourné avant ce texte

- `identite44.py` : le moteur de la clé 44 refait au bit près des courses publiées de la clé 43 — la recherche
  aléatoire, le système et les 32 recettes réglées de la graine 100, entiers ; le réseau court réglé de la graine 100,
  entier ; les trois premières générations du système de la graine 103 : 10 489 valeurs, aucun écart
  (`identite44*.json`). La diversité, nouvelle, s'ajoute aux sorties sans rien changer. Deux premiers lancements de la
  recherche aléatoire et du système, interrompus au bout d'une dizaine de minutes pour ajouter au moteur la protection
  du réseau livré d'ENS et d'ENSR, n'ont rien comparé.
- `essai44.py` : deux générations d'ENSR et d'ENS (avec les mesures finales) sur la graine 201, hors plan, pour
  vérifier que les bras et les mesures nouvelles tournent — une première fois avant l'ajout de cette protection, une
  seconde fois après ; seuls les durées, la forme des sorties et le nombre de lignées au deuxième passage du juge (24
  pour ENS, 32 pour ENSR) ont été affichés.
- `controle44.py --ecrire` puis `controle44.py` : les valeurs attendues du contrôle d'environnement, sur la graine 200
  (hors plan), écrites dans un fichier sans être affichées, puis le contrôle lui-même (« bon » quatre fois).
- `verdict44.py` et `ana44.py` ont été essayés sur des données de synthèse construites à partir des courses de la
  clé 43 (bras renommés, diversité tirée au hasard) — aucune donnée de la clé 44. `ana44.py`, l'analyse descriptive, a
  été écrit avant les courses et figure dans le manifeste.
- Rien du plan.

## Où et comment

Les tranches (E1 à E20, 12 courses chacune) tournent dans des sessions cloud que je lance depuis l'ordinateur de
Lucas, à sa demande, sur ses crédits ; chacune exécute une tranche telle qu'écrite (`cle44/CLAUDE.md`) et pousse ses
résultats sur sa branche du dépôt public de la clé 40 (https://github.com/Lucas2a63/cle40), où la clé 44 occupe le
dossier `cle44/` ; le bac à sable ne fait que lire ce dépôt. Rien n'agit hors de ce cadre : pas d'agent autonome, pas
d'accès réseau autre que numpy et git.

## Les limites connues d'avance

- Une seule architecture (le perceptron), un seul monde (fixe), une seule base d'images, des réseaux en numpy, sans
  augmentation de données, en descente de gradient simple.
- L'apport à l'ensemble est un critère parmi d'autres (la log-vraisemblance, en retirant un réseau, sur 200 images
  par génération, donc bruité) ; un autre critère — l'exactitude, un désaccord récompensé, une sélection moins
  fréquente — pourrait faire autrement. Le remplacement d'un quart de la population à chaque génération est celui du
  système.
- L'ambiguïté mêle la diversité des décisions et la netteté des probabilités ; le désaccord, qui ne dépend que des
  décisions, est rapporté à côté.
- Les mesures finales suivent la dernière sélection ; le grand réseau est repris de la clé 43 ; l'ensemble coûte à
  l'usage 32 fois un petit réseau.
- Les interactions entre ingrédients ne sont pas testées.
