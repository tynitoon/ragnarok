# Calibration indépendante de Pong — développement CPU

Le protocole est dans `ARC4_PONG_CALIBRATION_PROTOCOL.json`, écrit avant la
capture. Développement : seed 17011, 4 096 images brutes consécutives ; validation
réservée : 17012, à ouvrir après correction et scellement du décodeur. La palette
sémantique sert uniquement à vérifier la géométrie du juge. Le futur extracteur
générique ne l'utilise pas. Ces fixtures ne sont pas des données d'apprentissage.

## Baseline conservée

La projection initiale suit OCAtari v2.2.1, avec conversion explicite des octets
en entiers Python et suppression des boîtes de taille négative. Elle **échoue**
sur la raquette du joueur : 96,043 % des boîtes conjointement visibles sont à
au plus un pixel de chaque bord, contre 99 % requis. L'adversaire atteint 100 %,
la balle 99,514 %. Les accords de présence dépassent 99 % pour les trois.
Résultat : `craft_v6_out/v70_pong_calibration/development_baseline.json`.

La capture coûte 4 240 frames réelles : 4 096 pas, 72 de démarrage et 72 de reset.
Les séquences et le manifeste possèdent des empreintes SHA256. Les planches
`fixed_frames.png` sélectionnent des instants fixes, sans consulter les étiquettes.
La première image après reset présente une palette transitoire ; elle reste
comptée dans les erreurs de présence. À la frame 118, la balle masque une partie
de la raquette adverse : le vérificateur mesure l'étendue des pixels visibles,
sans supposer un rectangle intégralement rempli. Le journal de l'essai initial
qui rejetait cette occlusion est conservé.

La revue indépendante (seed 17, distincte des deux séquences) retrouve des
raquettes de 16 pixels, contre 15 dans la source, et un alignement vertical
pair pour le joueur. Une correction géométrique sera testée sur développement
avant ouverture de la validation ; les seuils seront conservés.

## Correction v1 sur développement, 9 septembre

Le joueur utilise `y = 2 * floor((RAM[51]-13)/2)` et une hauteur de 16 ;
l'adversaire garde `y = RAM[50]-15` avec une hauteur de 16 ; la balle utilise
`y = 2 * floor((RAM[54]-14)/2)` et une hauteur de 4. Les boîtes sont ensuite
intersectées avec le terrain. Ces conventions proviennent de la calibration,
pas d'une garantie fournie par OCAtari.

Sur 17011, les boîtes des deux raquettes sont maintenant exactes à 100 % ; celles
de la balle sont à un pixel près à 100 %. Accord de présence : joueur 99,951 %,
adversaire 100 %, balle 99,902 %. Les images de transition restent dans le
dénominateur. Résultat : `development_corrected.json`. La revue contradictoire
autorise le scellement de cette version avant 17012, sans modifier les seuils.

Les 13 tests de géométrie, lecture de fixtures, métriques et référence passent.
Le matching privilégie le nombre d'appariements valides avant la distance ;
le mouvement n'est pas réduit au déplacement net et l'historique repart à zéro
au reset. La référence apprend son fond modal depuis les seuls pixels passés,
garde les égalités stables et n'apprend pas des requêtes contrefactuelles.

Les 4 096 observations n'incluent pas les images internes des démarrages/reset :
leurs 144 frames sont facturées mais ne sont pas disponibles dans ces fixtures.
Le changement d'épisode empêche de les traiter comme une transition adjacente.

## Sources et portée

- [Décodeur OCAtari v2.2.1](https://github.com/k4ntz/OC_Atari/blob/v2.2.1/ocatari/ram/pong.py) :
  géométrie initiale et adresses RAM. Notice MIT dans `licenses/OCAtari-MIT.txt`.
- [AtariARI épinglé](https://github.com/mila-iqia/atari-representation-learning/blob/cc87a29ddce47b1800b9f82b4792284addbe00d1/atariari/benchmark/ram_annotations.py#L178-L185) :
  accord indépendant sur les adresses ; ses valeurs sont des coordonnées RAM brutes.
- [Pong ALE v0.12.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Pong.cpp#L24-L35) :
  le score net est joueur moins adversaire, le terminal survient à 21 points.
  [Les vies valent toujours zéro](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/games/supported/Pong.hpp#L69).
  Un point perdu ne constitue pas une baisse de vies ALE ; le gate devra expliciter
  les jeux où cette mesure est définie. La calibration géométrique ne valide pas
  le score, les décès, la compétence ou le transfert.

Le décodeur minimal est un port séparé, sans dépendance OCAtari à installer. Il
tourne dans l'environnement CPU 3.14 existant. Python 3.12 n'est donc pas requis
pour ce premier jeu ; la compatibilité des autres décodeurs reste à vérifier.
