# Freeway — calibration brute CPU

Le protocole `ARC4_FREEWAY_CALIBRATION_PROTOCOL.json` est établi avant la capture
17211. Validation 17212 fermée. Douze objets : deux poulets, dix voitures par
voie. Le deuxième poulet stationnaire participe à la présence et à la précision.
Il ne compte comme mobile que si ses positions changent dans l'historique.

Les coordonnées RAM et les couleurs viennent des modules MIT
[RAM Freeway](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/ram/freeway.py)
et [vision Freeway](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/vision/freeway.py)
d'OCAtari, commit fixé. Le checker indépendant cherche les pixels dans des zones
fixes, sans appeler le décodeur RAM ni recevoir ses boîtes. Aucun label oracle
n'entre dans la référence ou l'agent.

La baseline utilise les dimensions du port, clippe le terrain et conserve les
éventuelles erreurs de bouclage, d'occlusion ou d'animation. Les seuils de 99 %
de présence et de boîtes à un pixel seront appliqués à chaque type. Toutes les
frames restent incluses. Un résultat négatif sera conservé avant correction.
Pas de gate ni d'entraînement. L'absence de vies ALE dans ce jeu interdit de
présenter sa calibration comme validation d'une mesure de mort.

## Corrections de développement, avant validation

Baseline négative préservée dans `fc180c0` : le checker couleur des poulets
englobait les lignes jaunes du milieu de route. Le protocole correctif distinct
`ARC4_FREEWAY_CALIBRATION_V1_PROTOCOL.json` utilise des composantes huit-connexes
dans chaque moitié d'écran, dimensions 7×8 à trois pixels près, comme le détecteur
vision épinglé. Les marques de route d'un pixel de haut ne deviennent plus des
poulets. Les erreurs de fusion/occlusion restent comptées, sans position RAM
pour guider la recherche.

La projection des voitures reçoit le bouclage horizontal et la zone non rendue
observée à gauche, x<8, puis la hauteur visible huit. V1 conservée dans `45c78e3`
échoue encore sur la présence de la voie 4 : 98,7793 %. V2 corrige son décalage
empirique fixe de +1 pixel. Sur le développement, cette voiture parcourt les
160 positions, se déplace de −1 à chacune des 4095 transitions ; 3690 boîtes
non rognées montrent exactement ce décalage. Cela est compatible avec une phase
d'une frame ; l'ordre interne de dessin et de mise à jour RAM n'est pas prouvé.
Vingt-cinq coins visibles de largeur un ont une hauteur six au lieu de huit,
restent comptés et respectent la tolérance d'un pixel aux deux bords.

V2 passe les douze types. Poulet joueur : présence 4093/4096, boîtes 4083/4093
conformes (99,7557 %). Deuxième poulet exact. Voitures : toutes les boîtes à
un pixel près ; présence au moins 4072/4096 (99,4141 %). Les seuils et les
dénominateurs restent ceux fixés avant la capture. La séquence ne fait monter
le joueur que jusqu'à y=93 ; elle ne valide pas toute la traversée de la route.
Les trois absences du checker joueur sont des fragments visibles trop étroits
(frame 166 : 3×8 ; 167–168 : 3×8 et 1×2). Ces désaccords restent pénalisés ;
le checker ne constitue pas une annotation parfaite des occlusions.

## Traçabilité de l'ouverture réservée

Le candidat final, son protocole, la revue documentée et les dix empreintes de
`development_for_seal.json` ont été commités dans `b9d5a2b` avant 17212.
Le fichier `validation_seal.json` a ensuite été écrit à 02:17:28, avant la
capture. **Son ajout à l'index Git a échoué** (« Unable to write new index file »).
La commande PowerShell a poursuivi la capture malgré cet échec : le commit
séparé de ce fichier de métadonnées intervient donc après la capture, contrairement
à l'ordre prévu. Cette déviation est conservée, pas présentée comme un commit
de sceau antérieur. Le candidat et toutes ses empreintes étaient néanmoins
fixés dans `b9d5a2b` ; aucun code ni critère n'a changé après ouverture.
Les prochains lancements séparent les appels et contrôlent leur code de retour.

## Résultat réservé, candidat inchangé

17212 passe les douze types sur 4096 frames, 4228 facturées. Poulet joueur :
présence 4083/4096 (99,6826 %), boîtes 4068/4083 (99,6326 %). Deuxième poulet
exact. Toutes les boîtes de voitures sont à un pixel ; présence ≥99,4141 %.
Les erreurs du checker et de la projection restent incluses.

Les voitures suivent la même horloge déterministe dans les deux captures :
leurs statistiques sont identiques. Le nouveau seed et les actions produisent
une autre trajectoire du joueur, mais la validation ne constitue pas une nouvelle
réplication indépendante des trajectoires des voitures. La couverture et la
portée de cette calibration doivent être rapportées ainsi.
