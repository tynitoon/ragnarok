# ARC 4 v74 — protocole de mesure (fixé avant les lancements, 21 octobre 2026)

Exploratoire : obtenir une compétence réelle (battre le meilleur bouton constant), pas encore le
transfert. Le panel de conception du 21 octobre place cette compétence avant toute étape de transfert.

## Constats qui motivent v74 (mesures de développement, graines hors protocole)
1. v72/v73 (imitation de branches tenues 32 pas) restent au niveau du bouton constant sur Assault
   (v73 : 403 et 408 contre 407 ± 23), sous lui sur Demon Attack et Freeway.
2. Le gain mesuré en laboratoire est réel : sur Assault, le choix fait sur une seule branche
   déterministe vaut encore +0,79 événement par 104 pas quand il est réévalué 8 fois avec actions
   collantes (+1,69 quand il s'écarte du constant ; écart type par branche 0,27). Ce n'est pas du
   bruit de sélection (`scripts/check_branch_noise_v74.py`).
3. Le goulot est la distillation : hors ligne (1200 racines Assault, validation croisée par
   trajectoire), les réseaux captent 20 à 26 % du gain de l'oracle, et ce taux monte avec les
   données (13 % → 21 % → 26 % pour 240 → 480 → 960 racines). Écarts fins entre modèles non établis
   (revue : corrélation entre racines d'un même départ).
4. Tenir chaque décision 8 pas en jeu (la durée sur laquelle sa valeur a été mesurée) change le
   résultat : politique « étiquette » 410 → 462 contre 388 pour le constant (10 parties, graine 99,
   2000 pas de survie contre 591). Indice, pas preuve.

5. Attente écrite avant lancement : hors ligne sur Demon Attack (1200 racines, préparation v72), les
   réseaux ne captent rien du gain de l'oracle (+1,63 ; appris entre −0,10 et +0,05). Un négatif
   v74 sur Demon Attack est donc attendu. Essai rapide Freeway avec l'explorateur corrigé : démo de
   822 pas et 10 événements (vide en v73).

## Méthode v74 (amélioration par simulation, « rpi »)
Entrée v73 (signal de mort : compteur de vies OU « disparu 2 pas », coupé par l'agent s'il se perd
> 20 % ; suivi amorcé sur 5 observations). Corrections : l'explorateur utilise la perception validée
(défaut v73), un re-repérage par sonde remet le compteur de disparition à zéro, sans démo l'agent
part de la racine. Apprentissage :
1. l'agent trouve son meilleur bouton constant (chaque bouton tenu 104 pas depuis 16 états de démo,
   valeur = ses événements de score − 2 × ses morts) ;
2. 12 tours : racines visitées par la base (tour 0) puis par sa propre politique (DAgger), avec 15 %
   d'actions au hasard et sans racine en double ; depuis chaque racine, chaque bouton est tenu 8 pas
   puis le constant 96 pas ; étiquette = meilleur bouton (le constant en cas d'égalité) sur les 4
   premières vues ;
3. en jeu : politique gloutonne (température 0,05, ε 0,02), chaque décision tenue 8 pas.
Bras : `rpi` (96 racines par tour) et `rpi_big` (192 racines par tour : le levier des données).

## Jeux, graines, évaluation
Jeux : assault, freeway, demon_attack, name_this_game, carnival, breakout.
**Graines neuves** : entraînement 4 et 5 ; évaluation 13 et 14, 10 parties chacune, actions
collantes 0,25, 7000 pas au plus. Références recalculées sur 13 et 14 : chaque bouton constant,
aléatoire uniforme, aléatoire avec répétition, politique non entraînée via l'interface de l'agent.
Diagnostic enregistré à chaque préparation (jamais utilisé par l'agent) : valeur réelle, avec
actions collantes, des choix faits sur une branche déterministe (`branch_noise`).

## Critère, fixé maintenant (plus strict que v72/v73)
Un bras « dépasse le meilleur bouton constant » sur un jeu si **chacune** des deux graines
d'entraînement (20 parties) dépasse la moyenne du meilleur constant (20 parties) de plus de
2·sqrt(se_graine² + se_constant²), données complètes exigées. Réussite de l'étape : au moins 2 jeux
sur 6 pour un bras. Lecture secondaire : le test poolé de v72/v73. Aucun réglage par jeu après
observation ; résultats négatifs conservés.

## Ressources
Deux files en parallèle au plus (partage par jeu), garde de lancement à 14 Go libres, superviseur
RAM actif ; Tavern garde toujours 12 Go.
