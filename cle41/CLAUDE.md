# Consignes pour les sessions cloud — clé 41 (dossier `cle41/` du dépôt)

La clé 41 occupe le dossier `cle41/` de ce dépôt ; le `CLAUDE.md` de la racine est celui de la clé 40, qui est
terminée. **Pour une tâche de la clé 41, ne suis que ces consignes-ci, et fais toutes les commandes depuis le
dossier `cle41/`** (`cd cle41`).

Ce dossier contient une expérience **pré-enregistrée** (`PREENREGISTREMENT_41.md` ; empreintes du code, du
plan et des données dans `MANIFESTE41.sha256`). Chaque session cloud exécute **une tranche** du plan
(`plan41.json`), telle qu'écrite, et pousse ses résultats sur **sa propre branche de travail**. Rien
d'autre : pas d'analyse, pas de modification, pas d'initiative.

## Si le message dit « Tranche K » (K = G1…G7, A1…A6, D1…D8, W1…W6 ou M1…M3)

0. **Les données.** Il faut un dossier `donnees/` contenant `cifar32_a.npz`, `cifar32_b.npz` et
   `cifar32_t.npz`, dans `cle41/`. Ce dossier reste local et ne doit jamais être commité. Recolle-les à partir
   des morceaux qui sont à la racine du dépôt (ceux de la clé 40) :
   `cd cle41 && mkdir -p donnees && cat ../cifar32_a.npz.part* > donnees/cifar32_a.npz && cat ../cifar32_b.npz.part* > donnees/cifar32_b.npz && cat ../cifar32_t.npz.part* > donnees/cifar32_t.npz`
1. **Vérifier que rien n'a bougé** : `sha256sum -c MANIFESTE41.sha256`. Si une ligne échoue, arrête-toi
   et signale-le, sans rien lancer.
2. **Pour une tranche W ou M seulement** : le message donne l'empreinte de `balayage41_choix.json`. Vérifie
   `sha256sum balayage41_choix.json` ; si le fichier manque ou si l'empreinte diffère, arrête-toi et
   signale-le, sans rien lancer. (Les tranches G, A et D n'en ont pas besoin.)
3. **numpy** : `python3 -c "import numpy; print(numpy.__version__)"`. S'il manque :
   `pip install numpy` (si pip refuse : `pip install --break-system-packages numpy`).
4. **Contrôle** : `python3 controle41.py` (environ trois minutes) doit afficher « bon » quatre fois. S'il
   affiche un écart, arrête-toi et signale l'écart, sans rien lancer.
5. **Reprise éventuelle** : `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` puis
   `python3 rassembler41.py K`. Si une session précédente avait commencé la tranche K, ses courses finies
   sont récupérées et ne seront pas refaites.
6. **Lancer en arrière-plan** :
   `nohup python3 lancer41.py --tranche K > journal_K.log 2>&1 &`
   puis, une minute plus tard, une première sauvegarde (elle montre la machine et le nombre de processus) :
   `git add -f journal_K.log && git commit -m "tranche K : lancement" && git push -u origin HEAD`
7. **Surveiller sans laisser la session s'arrêter.** Une tranche dure d'une demi-heure à trois heures. Vérifie
   l'avancement toutes les dix minutes environ (`tail -n 3 journal_K.log`), avec une attente en arrière-plan
   ou l'outil Monitor si une longue commande `sleep` est refusée. Dès qu'une course est finie, puis à chaque
   vérification où le journal a avancé, sauvegarde sur ta branche de travail :
   `git add -f resultats/tranche_K.json journal_K.log && git commit -m "tranche K : partiel" && git push -u origin HEAD`
8. **Si le calcul s'est arrêté** (`pgrep -f "lancer41.py --tranche"` ne renvoie rien) sans que le journal
   n'affiche `TRANCHE K TERMINÉE`, relance la commande de l'étape 6 : elle reprend là où elle s'était
   arrêtée, sans refaire les courses finies.
9. **À la fin** (`TRANCHE K TERMINÉE` dans le journal) : dernier commit et `git push -u origin HEAD`, puis
   réponds par les cinq dernières lignes du journal et le nom de ta branche. C'est terminé.

## Ce que tu ne fais jamais

- Modifier un fichier suivi du dépôt (code, plan, pré-enregistrement, ces consignes) — même pour
  « corriger » ou « améliorer » —, ni rien hors du dossier `cle41/`. Si quelque chose ne marche pas,
  arrête-toi et décris l'erreur.
- Analyser, résumer ou commenter les résultats : le verdict se fait ailleurs, selon la règle écrite avant
  les courses.
- Ouvrir une pull request, pousser sur `main` ou sur une autre branche que la tienne.
- Tout accès réseau autre que `pip install numpy` et `git` sur ce dépôt.
- Lancer une autre tranche que celle du message, ou plus d'une tranche à la fois.
