# Revue indépendante — extension de l'instrument ARC 4

Revue en lecture seule des sources ALE 0.12.1 et OCAtari, sans ouverture de ROM
hors Pong, ni calcul GPU. Les cinq jeux sont ceux de développement déjà proposés
pour le gate 0 ; aucun jeu de confirmation n'a été inspecté.

## Coûts à vérifier avant activation

Le reset général exécute 60 NOOP et deux soft-resets de quatre frames.
`pressSelect(n)` exécute n+1 frames mais n'incrémente le getter ALE que d'une unité.
Les cinq jeux n'ont pas de `startingActions` supplémentaires. Sources :
[reset](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/environment/stella_environment.cpp#L101-L130),
[pressSelect](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/environment/stella_environment.cpp#L257-L266),
[défauts RomSettings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/RomSettings.cpp#L52-L65).

| Jeu | Coût exact | Sélection du mode public 0 | MD5 ROM reconnu |
|---|---|---|---|
| Breakout | 72 + 2k | RAM[50]=0 | f34f08e5eb96e500e851a80be3277a56 |
| Freeway | 72 + 2k | RAM[0]=0 | 8e0ab801b1705a740b476b7f588c6d16 |
| Seaquest | 68 | Aucune sélection supplémentaire | 240bfbac5163af4df5ae713985386f92 |
| MsPacman | 72 + 11k | RAM[25]=1 et RAM[33]=1 | 87e79cd41ce136fd4f72cc6e2c161bee |
| Asteroids | 72 + 3k | RAM[0]=0 | ccbd36746ed4525821a8083b0d6d2c2c |

k est le nombre d'itérations de sélection : delta du getter de frames lors
d'un reset, getter initial après chargement. Les boucles sources ne fixent pas
de plafond explicite. La formule donne une comptabilité après exécution, pas
une borne suffisante pour pré-admettre le prochain reset. Mesurer k sur les
ROMs de développement puis documenter les hypothèses avant leur activation.

Sources des sélections : [Breakout](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Breakout.cpp#L117-L133),
[Freeway](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Freeway.cpp#L99-L115),
[MsPacman](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/MsPacman.cpp#L118-L152),
[Asteroids](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Asteroids.cpp#L127-L143).

## Événements et dénominateurs

Pong et Freeway retournent constamment zéro pour les vies ALE. La clause de F1
décès sur cinq des six jeux n'est donc pas évaluable. Ne pas attribuer F1=1
à une série sans événements positifs ; distinguer points perdus, collisions,
pertes de vies, terminal et reset. Cette correction de portée doit précéder le
gel, indépendamment des performances du détecteur. Sources :
[Pong lives](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Pong.hpp#L69),
[Freeway lives](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Freeway.hpp#L74).

Les décodeurs [OCAtari épinglés](https://github.com/k4ntz/OC_Atari/tree/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/ram)
existent mais ne partagent pas les conventions géométriques de Pong :

- Breakout comporte des blocs statiques ; séparer objets visibles et objets mobiles.
- Freeway comporte deux poulets et dix voitures ; les collisions ne retirent pas
  de vies ALE.
- Seaquest distingue des phases de crash et des objets partageant des adresses.
  Les vies HUD n'incluent pas la vie courante qu'ALE ajoute.
- MsPacman comporte fantômes, fruits, pilules et changement de plateau ; le HUD
  vers y187 interdit de reprendre le rectangle d'évaluation de Pong.
- Asteroids encode coordonnées, orientation et tailles ; le joueur peut
  disparaître. Une projection affine unique ne suffit pas.

Chaque jeu demandera sa calibration écran/RAM indépendante, avec visibilité,
géométrie, HUD et événements définis avant sa vérification réservée. L'interface
pixels et les snapshots sont réutilisables ; les étiquettes et les coûts doivent
être vérifiés jeu par jeu.
