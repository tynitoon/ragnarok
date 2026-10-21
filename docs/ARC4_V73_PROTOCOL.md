# ARC 4 v73 — protocole de mesure (fixé avant les lancements, 21 octobre 2026)

Exploratoire, comme v72 : obtenir une compétence réelle sur un jeu, pas encore le transfert.

## Ce qui change par rapport à v72
1. **Signal de mort lu à l'écran.** v72 ne pénalisait que la fin de partie sur 5 jeux sur 6 : aucun
   compteur de vies trouvé hors Breakout (revue contradictoire du 21 octobre). v73 compte une vie
   perdue quand le compteur de vies de l'agent change OU quand son propre suivi ne le trouve nulle
   part pendant 2 pas de suite. L'agent coupe ce second signal s'il se perd lui-même plus de 20 % du
   temps en jeu ordinaire (mesuré à l'entrée, sur ses propres images).
2. **Suivi de soi amorcé sur plusieurs observations.** Quand les premiers écrans ignorent les
   boutons, la sonde au départ ne trouvait personne et le suivi n'était jamais amorcé (Name This
   Game perdu 100 % du temps en v72). L'agent joue et sonde jusqu'à 5 observations ; sa taille est
   la médiane de ces observations.
3. Corrections mineures de la revue : vues précédentes stockées avant mise à jour, lecteur de
   compteurs transporté dans chaque racine, mêmes nombres aléatoires pour les branches d'une racine.

## Pré-contrôle (côté évaluateur, graine 5, hors graines d'entraînement et d'évaluation)
`scripts/check_death_v73.py`, 8000 pas de jeu aléatoire avec répétition par jeu ; une alerte compte
si une vraie perte de vie (`env.lives()`, jamais vu par l'agent) tombe dans [-30, +60] pas.
Résultats (`craft_v6_out/v73/death_check.json`, entrée v73), signal « compteur de vies OU disparu 2 pas » :

| jeu | vraies pertes (dont fins de partie) | alertes | précision | rappel | perdu de vue |
|---|---|---|---|---|---|
| assault | 30 | 30 | 1,00 | 1,00 | 2,8 % |
| demon_attack | 28 | 29 | 0,97 | 1,00 | 5,6 % |
| name_this_game | 15 | 11 | 1,00 | 0,73 | 2,7 % |
| breakout (compteur de vies) | 90 | 72 | 1,00 | 0,92 | 0 % |
| freeway (pas de vies) | 3 | 0 | — | — | 0,1 % |
| carnival (pas de vies) | 8 | 0 | — | — | 0 % |

Avant correction de l'amorçage, Name This Game était perdu de vue 100 % du temps (v72 compris).
k = 2 est retenu pour tous les jeux (k = 1 donne des fausses alertes sur Freeway, k ≥ 8 manque les
morts d'Assault) ; ce choix est fait sur la graine 5 avant toute mesure v73.

## Bras, graines
Jeux : assault, freeway, demon_attack, name_this_game, carnival, breakout.
**Graines neuves** : entraînement 2 et 3 ; évaluation 11 et 12 (10 parties chacune, actions
collantes 0,25). Préparation commune en cache (entrée + exploration 1500 itérations), puis :
- `bri` : branches tenues 32 pas, sans continuation, 24 racines par tour, 12 tours ;
- `bri_big` : idem avec 64 racines par tour.
Références recalculées sur les graines 11 et 12 : chaque bouton constant, aléatoire uniforme,
aléatoire avec répétition, politique non entraînée via l'interface de l'agent.

## Critère, fixé maintenant
Identique à v72 : un bras « dépasse le meilleur bouton constant » sur un jeu si sa moyenne poolée
(2 graines × 20 parties) dépasse celle du meilleur constant (20 parties) de plus de
2·sqrt(se_m² + se_c²), données complètes exigées. Réussite : au moins 2 jeux sur 6 pour un bras.
Lecture secondaire déclarée d'avance : « robuste » si, en plus, la moyenne de chaque graine
d'entraînement dépasse celle du meilleur constant. Comparaison descriptive avec v72 (mêmes jeux,
autres graines). Aucun réglage par jeu après observation ; résultats négatifs conservés.

## Ressources
Un seul processus à la fois, garde de lancement à 14 Go libres, superviseur RAM actif ; Tavern
garde toujours 12 Go.
