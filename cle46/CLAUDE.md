# Consignes pour les sessions cloud — clé 46 (dossier `cle46/` du dépôt)

La clé 46 occupe le dossier `cle46/` de ce dépôt ; le `CLAUDE.md` de la racine est celui de la clé 40, ceux de
`cle41/` à `cle45/` ceux des clés 41 à 45, qui sont terminées. **Pour une tâche de la clé 46, ne suis que ces
consignes-ci, et fais toutes les commandes depuis le dossier `cle46/`** (`cd cle46`).

Ce dossier contient une expérience **pré-enregistrée** (`PREENREGISTREMENT_46.md` ; empreintes du code, du plan, des
réglages et des données dans `MANIFESTE46.sha256`). Chaque session cloud exécute **une tranche** du plan
(`plan46.json`), telle qu'écrite, et pousse ses résultats sur **sa propre branche de travail**. Rien d'autre : pas
d'analyse, pas de modification, pas d'initiative.

## Si le message dit « Tranche K » (K = H1…H12)

0. **Les données.** Il faut un dossier `donnees/` contenant `cifar32_a.npz`, `cifar32_b.npz` et `cifar32_t.npz`,
   dans `cle46/`. Ce dossier reste local et ne doit jamais être commité. Recolle-les à partir des morceaux qui sont à
   la racine du dépôt (ceux de la clé 40) :
   `cd cle46 && mkdir -p donnees && cat ../cifar32_a.npz.part* > donnees/cifar32_a.npz && cat ../cifar32_b.npz.part* > donnees/cifar32_b.npz && cat ../cifar32_t.npz.part* > donnees/cifar32_t.npz`
1. **Vérifier que rien n'a bougé** : `sha256sum -c MANIFESTE46.sha256`. Si une ligne échoue, arrête-toi et
   signale-le, sans rien lancer.
2. **numpy** : `python3 -c "import numpy; print(numpy.__version__)"`. S'il manque : `pip install numpy` (si pip
   refuse : `pip install --break-system-packages numpy`).
3. **Contrôle** : `python3 controle46.py` (environ un quart d'heure) doit afficher « bon » trois fois. S'il affiche un
   écart, arrête-toi et signale l'écart, sans rien lancer.
4. **Reprise éventuelle** : `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` puis
   `python3 rassembler46.py K`. Si une session précédente avait commencé la tranche K, ses courses finies sont
   récupérées et ne seront pas refaites.
5. **Lancer en arrière-plan** :
   `nohup python3 lancer46.py --tranche K > journal_K.log 2>&1 &`
   puis, une minute plus tard, une première sauvegarde (elle montre la machine et le nombre de processus) :
   `git add -f journal_K.log && git commit -m "clé 46, tranche K : lancement" && git push -u origin HEAD`
6. **Surveiller sans laisser la session s'arrêter.** Une tranche (dix courses sur quatre processus) dure
   environ cinq heures (jusqu'à huit sur une machine lente) : c'est long, et c'est normal. **Ne termine pas ton tour tant
   que la tranche tourne** : une session inactive est arrêtée et son calcul perdu. Vérifie l'avancement toutes les dix minutes environ (`tail -n 3 journal_K.log`), avec une attente en arrière-plan ou
   l'outil Monitor si une longue commande `sleep` est refusée. Dès qu'une course est finie, puis à chaque vérification
   où le journal a avancé, sauvegarde sur ta branche de travail :
   `git add -f resultats/tranche_K.json journal_K.log && git commit -m "clé 46, tranche K : partiel" && git push -u origin HEAD`
7. **Si le calcul s'est arrêté** (`pgrep -f "lancer46.py --tranche"` ne renvoie rien) sans que le journal n'affiche
   `TRANCHE K TERMINÉE`, relance la commande de l'étape 5 : elle reprend là où elle s'était arrêtée, sans refaire les
   courses finies.
8. **À la fin** (`TRANCHE K TERMINÉE` dans le journal) : dernier commit et `git push -u origin HEAD`, puis réponds
   par les cinq dernières lignes du journal et le nom de ta branche. C'est terminé.

## Ce que tu ne fais jamais

- Modifier un fichier suivi du dépôt (code, plan, pré-enregistrement, ces consignes) — même pour « corriger » ou
  « améliorer » —, ni rien hors du dossier `cle46/`. Si quelque chose ne marche pas, arrête-toi et décris l'erreur.
- Analyser, résumer ou commenter les résultats : le verdict se fait ailleurs, selon la règle écrite avant les
  courses.
- Ouvrir une pull request, pousser sur `main` ou sur une autre branche que la tienne.
- Tout accès réseau autre que `pip install numpy` et `git` sur ce dépôt.
- Lancer une autre tranche que celle du message, ou plus d'une tranche à la fois.
