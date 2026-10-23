# ARC 4 v84 — « Porter les questions, pas les réponses » (protocole, version française)

Figé le 23 octobre 2026, avant tout calcul de v84. **Le texte qui fait foi est l'anglais :
`docs/ARC4_V84_PROTOCOL_EN.md`** (plan final du panel de conception, workflows `wf_55fab7f2-81d`
propositions, `wf_c73b5d24-865` juges, `wf_a2b93ac1-81e` synthèse, audits et version finale ; copies dans
`docs/panels/`). Ce document en est le résumé français et s'y soumet en cas de différence.

## En-tête de toute affirmation
« Toute affirmation de v84 porte sur une RECHERCHE AVEC RETOURS EN ARRIÈRE FACTURÉS (savestates, 60 trames
par chargement, chaque trame émulée facturée) et utilise les SENS accordés le 22 octobre 2026 (hausses du
score, nombre de vies). Ce n'est pas du jeu en direct sans savestates. »

## Ce que fait v84, dans l'ordre
1. **G0, l'horloge du savoir** (aucune image d'émulateur, sur les 82 journaux A1 de v83). Précédé d'un
   contrôle « planté » qui vérifie que la méthode détecte une structure partagée. Question : un a priori
   de covariance à moyenne nulle Σ, appris sur 40 autres jeux, fait-il atteindre plus vite (en images
   de la même expérience) au prédicteur BON/MAUVAIS d'un jeu tenu à l'écart sa propre compétence de fin
   de flux ? Il doit battre, jeu par jeu, le meilleur de trois témoins : F (isotrope réglé), P (placebo à
   étiquettes permutées, même spectre), U (a priori par ACP des entrées, même spectre), y compris sur la
   fin de courbe, et le plafond intra-jeu K\* doit montrer qu'il y a de la place. Moyenne nulle : aucune
   avance au départ n'est possible (certificat d'identité des préfixes). Spécification complète :
   partie A du texte anglais.
2. **V2, un explorateur neuf réparé**, construit en parallèle : sans sondes de « moi », cases par code
   d'image, il continue après une vie perdue, joue les quatre processus de boutons de la barre. Porte
   exploratoire G1a (ne tue qu'une construction cassée), puis porte G1b « place » sur les 41 jeux ouverts,
   au niveau d'A1 (au moins 13 des 35 jeux de développement).
3. **Au plus un objet transporté passe en test réel** : Σ si G0 passe, sinon le « sens de la recherche »
   s'il passe sa propre horloge G0-S. Contre V2 seul, les réglages par défaut T, le meilleur processus
   fixe S, deux placebos P et U, une dose et le plafond K\*. Le lot scellé (24 jeux) n'est ouvert qu'une
   fois ; le second tour (9 jeux) seulement en complément ou en option M.

## Graines (vérifiées le 23 octobre)
Bloc 9700-9749 (table §9 du texte anglais). Vérification : `git ls-files` des dépôts `ragnarok` et
`ragnarok_v83` (environnements virtuels exclus), motif `97[0-4][0-9]` : aucune occurrence dans un contexte
de graine ; les seuls nombres trouvés sont des compteurs d'images (`craft_v6_out/v70_action_signal_survey`)
et des mesures de profilage (`craft_v6_out/v70_breakout_pooled`).

## Contamination (complément au §3.4 anglais)
Le chef de projet (Claude) a lui aussi vu la ligne de journal « good alien False » de l'analyse
exploratoire v83 avant ce gel (en vérifiant que le calcul tournait). Cette ligne est un drapeau de
convergence de l'optimiseur pour un ajustement, pas un résultat de prédiction. Rien d'autre n'a été lu.
Le résultat exploratoire reste scellé jusqu'au commit de `g0_verdict.json`.

## Chances (texte anglais §12)
G0 passe : ≈ 0,2. V2 a de la place sur ≥ 13/35 : ≈ 0,21. Positif scellé partiel ou mieux : ≈ 0,5 %.
Issue la plus probable : un négatif propre de G0 en quelques jours, une carte des coûts d'un explorateur
sans sondes, et tous les témoins prêts.

## Décision réservée à Jérémie
Option M (texte anglais §16) : utiliser les 9 jeux du second tour si G0 passe mais que la voie en direct
s'arrête. Demandée seulement si ce cas se présente.
