# ARC 4 v72 — protocole de mesure (fixé avant les lancements, 21 octobre 2026)

Exploratoire : il s'agit d'obtenir une compétence réelle sur un jeu, pas encore le transfert.
Corrections appliquées avant toute mesure : D1 à D5 de `docs/ARC4_V71_DIAGNOSIS.md`
(fit masqué, adaptateur sans perte ni fusion de boutons, démo rejouée depuis la racine,
même perception au rejeu qu'en direct, auto-imitation par branches).

## Jeux et bras
Jeux : assault, freeway, demon_attack, name_this_game, carnival, breakout.
Pour chaque jeu, graines d'entraînement 0 et 1 : une préparation commune en cache (entrée +
exploration 1500 itérations), puis chaque méthode apprend depuis ce cache (politique neuve, graine
torch 1000+graine, 12 tours) :
- `bri` : branches tenues 32 pas, sans continuation (règle testée par le diagnostic) ;
- `segments` : ancienne règle (comparaison) ;
- `bri_cont` (assault, carnival seulement) : branches tenues 8 pas puis 24 pas de politique.

## Évaluation
En direct, actions collantes 0,25, graines d'évaluation 7 et 8, 10 parties chacune (20 parties par
bras et graine d'entraînement). Références sur les mêmes 20 parties : chaque bouton constant,
aléatoire uniforme, aléatoire avec répétition (0,95), politique non entraînée passant par
l'interface de l'agent. Vrai score lu côté évaluateur uniquement.

## Critère, fixé maintenant
Une méthode « dépasse le meilleur bouton constant » sur un jeu si sa moyenne poolée (2 graines
d'entraînement × 20 parties) dépasse celle du meilleur bouton constant (20 parties) de plus de
2 erreurs types combinées : Δ > 2·sqrt(se_m² + se_c²). Réussite de l'étape : au moins 2 jeux sur 6.
Sinon, résultat négatif conservé tel quel. Aucun réglage par jeu après observation.

## Ressources
Un seul processus à la fois ; garde de lancement à 14 Go libres ; superviseur RAM actif
(`scripts/ram_supervisor.py`) ; la session Tavern garde toujours 12 Go.
