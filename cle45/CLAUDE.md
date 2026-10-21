# Consignes pour les sessions cloud — clé 45 (dossier `cle45/` du dépôt)

La clé 45 occupe le dossier `cle45/` de ce dépôt ; le `CLAUDE.md` de la racine est celui de la clé 40, ceux de
`cle41/` à `cle44/` ceux des clés 41 à 44, qui sont terminées. **Pour une tâche de la clé 45, ne suis que ces
consignes-ci, et fais toutes les commandes depuis le dossier `cle45/`** (`cd cle45`).

Ce dossier contient une expérience **pré-enregistrée** (`PREENREGISTREMENT_45.md` ; empreintes du code, du plan, des
réglages et des données dans `MANIFESTE45.sha256`). Chaque session cloud exécute **une tranche** du plan
(`plan45.json`), telle qu'écrite, et pousse ses résultats sur **sa propre branche de travail**. Rien d'autre : pas
d'analyse, pas de modification, pas d'initiative.

## Si le message dit « Tranche K » (K = F1…F12)

0. **Les données.** Il faut un dossier `donnees/` contenant `cifar32_a.npz`, `cifar32_b.npz` et `cifar32_t.npz`,
   dans `cle45/`. Ce dossier reste local et ne doit jamais être commité. Recolle-les à partir des morceaux qui sont à
   la racine du dépôt (ceux de la clé 40) :
   `cd cle45 && mkdir -p donnees && cat ../cifar32_a.npz.part* > donnees/cifar32_a.npz && cat ../cifar32_b.npz.part* > donnees/cifar32_b.npz && cat ../cifar32_t.npz.part* > donnees/cifar32_t.npz`
1. **Vérifier que rien n'a bougé** : `sha256sum -c MANIFESTE45.sha256`. Si une ligne échoue, arrête-toi et
   signale-le, sans rien lancer.
2. **numpy** : `python3 -c "import numpy; print(numpy.__version__)"`. S'il manque : `pip install numpy` (si pip
   refuse : `pip install --break-system-packages numpy`).
3. **Contrôle** : `python3 controle45.py` (environ trois minutes) doit afficher « bon » deux fois. S'il affiche un
   écart, arrête-toi et signale l'écart, sans rien lancer.
4. **Reprise éventuelle** : `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` puis
   `python3 rassembler45.py K`. Si une session précédente avait commencé la tranche K, ses courses finies sont
   récupérées et ne seront pas refaites.
5. **Lancer en arrière-plan** :
   `nohup python3 lancer45.py --tranche K > journal_K.log 2>&1 &`
   puis, une minute plus tard, une première sauvegarde (elle montre la machine et le nombre de processus) :
   `git add -f journal_K.log && git commit -m "clé 45, tranche K : lancement" && git push -u origin HEAD`
6. **Surveiller sans laisser la session s'arrêter.** Une tranche (quatre courses en parallèle) dure environ
   trois quarts d'heure (plus sur une machine lente). Vérifie l'avancement toutes les dix minutes environ (`tail -n 3 journal_K.log`), avec une attente en arrière-plan ou
   l'outil Monitor si une longue commande `sleep` est refusée. Dès qu'une course est finie, puis à chaque vérification
   où le journal a avancé, sauvegarde sur ta branche de travail :
   `git add -f resultats/tranche_K.json journal_K.log && git commit -m "clé 45, tranche K : partiel" && git push -u origin HEAD`
7. **Si le calcul s'est arrêté** (`pgrep -f "lancer45.py --tranche"` ne renvoie rien) sans que le journal n'affiche
   `TRANCHE K TERMINÉE`, relance la commande de l'étape 5 : elle reprend là où elle s'était arrêtée, sans refaire les
   courses finies.
8. **À la fin** (`TRANCHE K TERMINÉE` dans le journal) : dernier commit et `git push -u origin HEAD`, puis réponds
   par les cinq dernières lignes du journal et le nom de ta branche. C'est terminé.

## Ce que tu ne fais jamais

- Modifier un fichier suivi du dépôt (code, plan, pré-enregistrement, ces consignes) — même pour « corriger » ou
  « améliorer » —, ni rien hors du dossier `cle45/`. Si quelque chose ne marche pas, arrête-toi et décris l'erreur.
- Analyser, résumer ou commenter les résultats : le verdict se fait ailleurs, selon la règle écrite avant les
  courses.
- Ouvrir une pull request, pousser sur `main` ou sur une autre branche que la tienne.
- Tout accès réseau autre que `pip install numpy` et `git` sur ce dépôt.
- Lancer une autre tranche que celle du message, ou plus d'une tranche à la fois.
