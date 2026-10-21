# ARC 4 v75 — tests hors ligne de la distillation (seuils fixés avant tout calcul, 21 octobre 2026)

Contexte : panel « goulot de distillation » du 21 octobre (4 concepteurs + 1 juge). Constats vérifiés
par le juge sur les fichiers `craft_v6_out/v74/qdata_*.npz` (sans calcul Python) :
- Assault : environ deux tiers du gain hors ligne v74 (+0,32) viennent d'un biais indépendant de
  l'écran (« une pause de 8 pas », +0,20) ; 9,4 % des racines ont un jumeau identique dans un autre
  pli (fuite).
- La jauge de chaleur d'Assault (lignes 192-199) est hors de la vue globale (le regroupement 32×32
  ne garde que les lignes 0-191).
- Demon Attack : les valeurs des boutons de directions différentes sont presque indépendantes
  (corrélations 0,02-0,29) ; l'« oracle » +1,63 est proche du maximum attendu de 3 tirages bruités.

Mesure : `scripts/offline_v75.py` — plis regroupant les trajectoires qui partagent un écran
(`link`, ou `sid` si présent), gain **au-dessus de la meilleure déviation constante** (`const_dev`,
choisie sur les plis d'entraînement), intervalle 95 % par bootstrap par groupes, graines torch 0, 1, 2.
Comparaisons appariées entre exécutions : `scripts/paired_v75.py`.

## T1 — mesure honnête (données existantes)
`conv_ce`, `conv_q` sur Assault et Demon Attack (`link`), Freeway (`sid`).
Lecture fixée : si l'intervalle de `conv_ce` au-dessus de `const_dev` contient 0 sur Assault, on
consigne « le 26 % de v74 était le biais de pause et la fuite ». Tous les tests suivants utilisent
cette mesure.

## T2 — défauts d'entrée
- Assault `FULL_H=1` (vue globale pleine hauteur) : réussi si `conv_ce` gagne ≥ +0,05 apparié sur
  T1 (intervalle > 0).
- Demon Attack `CLEAN=1` (pixels égaux à la couleur dominante de leur ligne de fond retirés) :
  réussi si ≥ +0,10 apparié (intervalle > 0).
- Freeway `FULL_H=1 CLEAN=1` : pas pire que T1 de plus de 0,03.

## T3 — pertes conscientes de la décision
`conv_set` (n'importe quel meilleur bouton est juste) et `conv_regret` (regret attendu), avec les
interrupteurs d'entrée retenus en T2. Adopté si ≥ +0,04 apparié sur `conv_ce` dans un jeu, sans jeu
pire de plus de 0,03 ; ou pour la calibration si à −0,02 de `conv_ce` en déviant au plus autant que
l'oracle.

Résultats négatifs conservés ; aucun seuil modifié après observation. Ces tests sont de
développement (données de graines déjà vues) ; seule une mesure en direct sous protocole figé peut
établir une compétence.
