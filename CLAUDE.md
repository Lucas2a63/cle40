# Consignes pour les sessions cloud — clé 40

Ce dépôt contient une expérience **pré-enregistrée** (`PREENREGISTREMENT_40.md` ; empreintes du code, du
plan et des données dans `MANIFESTE40.sha256`). Chaque session cloud exécute **une tranche** du plan
(`plan40.json`), telle qu'écrite, et pousse ses résultats sur **sa propre branche de travail**. Rien
d'autre : pas d'analyse, pas de modification, pas d'initiative.

## Si le message dit « Tranche K » (K = B1…B3, P1…P8 ou R1…R8)

0. **Les données.** Il faut un dossier `donnees/` contenant `cifar32_a.npz`, `cifar32_b.npz` et
   `cifar32_t.npz`. Ce dossier reste local et ne doit jamais être commité. Recolle-les à partir des morceaux
   du dépôt :
   `mkdir -p donnees && cat cifar32_a.npz.part* > donnees/cifar32_a.npz && cat cifar32_b.npz.part* > donnees/cifar32_b.npz && cat cifar32_t.npz.part* > donnees/cifar32_t.npz`
1. **Vérifier que rien n'a bougé** : `sha256sum -c MANIFESTE40.sha256`. Si une ligne échoue, arrête-toi
   et signale-le, sans rien lancer.
2. **Pour une tranche R seulement** : le message donne l'empreinte de `balayage40_choix.json`. Vérifie
   `sha256sum balayage40_choix.json` ; si le fichier manque ou si l'empreinte diffère, arrête-toi et
   signale-le, sans rien lancer. (Les tranches B et P n'en ont pas besoin.)
3. **numpy** : `python3 -c "import numpy; print(numpy.__version__)"`. S'il manque :
   `pip install numpy` (si pip refuse : `pip install --break-system-packages numpy`).
4. **Contrôle** : `python3 controle40.py` (environ cinq minutes) doit afficher « bon » quatre fois. S'il
   affiche un écart, arrête-toi et signale l'écart, sans rien lancer.
5. **Reprise éventuelle** : `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` puis
   `python3 rassembler40.py K`. Si une session précédente avait commencé la tranche K, ses courses finies
   sont récupérées et ne seront pas refaites.
6. **Lancer en arrière-plan** :
   `nohup python3 lancer40.py --tranche K > journal_K.log 2>&1 &`
   puis, une minute plus tard, une première sauvegarde (elle montre la machine et le nombre de processus) :
   `sleep 60; git add -f journal_K.log && git commit -m "tranche K : lancement" && git push -u origin HEAD`
7. **Surveiller sans rendre la main.** Répète, un appel à la fois, avec un délai d'outil de 600 000 ms :
   `sleep 540; tail -n 3 journal_K.log`.
   **Ne termine pas ton tour tant que la tranche tourne** : une session inactive est arrêtée et son calcul
   perdu. Une tranche peut durer de une à sept heures. Dès qu'une course est finie, puis toutes les trois
   vérifications environ, sauvegarde sur ta branche de travail :
   `git add -f resultats/tranche_K.json journal_K.log && git commit -m "tranche K : partiel" && git push -u origin HEAD`
8. **Si le calcul s'est arrêté** (`pgrep -f "lancer40.py --tranche"` ne renvoie rien) sans que le journal
   n'affiche `TRANCHE K TERMINÉE`, relance la commande de l'étape 6 : elle reprend là où elle s'était
   arrêtée, sans refaire les courses finies.
9. **À la fin** (`TRANCHE K TERMINÉE` dans le journal) : dernier commit et `git push -u origin HEAD`, puis
   réponds par les cinq dernières lignes du journal et le nom de ta branche. C'est terminé.

## Ce que tu ne fais jamais

- Modifier un fichier suivi du dépôt (code, plan, pré-enregistrement, ces consignes) — même pour
  « corriger » ou « améliorer ». Si quelque chose ne marche pas, arrête-toi et décris l'erreur.
- Analyser, résumer ou commenter les résultats : le verdict se fait ailleurs, selon la règle écrite avant
  les courses.
- Ouvrir une pull request, pousser sur `main` ou sur une autre branche que la tienne.
- Tout accès réseau autre que `pip install numpy` et `git` sur ce dépôt.
- Lancer une autre tranche que celle du message, ou plus d'une tranche à la fois.
