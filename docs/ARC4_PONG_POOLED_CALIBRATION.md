# Calibration des observations max-poolées de Pong

Protocole distinct fixé au commit `3c76a4b`, avant capture des seeds 17021
(développement) et 17022 (réservée, encore fermée à ce stade). Le décodeur RAM
brut reste celui scellé précédemment ; l'adaptateur temporel est séparé.

L'observation est le maximum canal par canal des deux dernières images réelles.
Une couleur d'objet peut devenir hybride ; un pixel de raquette peut disparaître
sous une balle blanche. Le vérificateur sémantique analyse donc les deux images
sources séparément et conserve les contributions qui survivent au maximum.
Il ne reçoit aucune RAM et n'utilise pas l'extracteur de référence.

L'adaptateur RAM candidat prend l'enveloppe des deux boîtes brutes de chaque
type. Une entité peut avoir un support visible disjoint dans la fenêtre ; on
mesure son enveloppe, pas la position de la dernière frame seule.

## Développement

2 048 fenêtres réellement émises à action repeat4, seed17021. Chaque observation
est vérifiée byte à byte contre le maximum des deux RGB enregistrés. Le tap RAM
encadre le dernier `act` réel ; aucun pas n'intervient entre la RAM et son image.
Au reset, les images viennent du nouvel épisode, jamais de la fin précédente.

Coût : 8 335 frames, dont 8 191 frames de jeu et 144 de démarrage/reset ; deux
épisodes et une répétition tronquée par la fin de partie. Les deux séquences RAM
et les deux RGB sont conservées avec empreintes dans des shards compressés.

Résultat `craft_v6_out/v70_pong_pooled/development_result.json` : présence 100 %
pour les trois types ; boîtes des deux raquettes exactes à 100 % ; balle à un
pixel près à 100 %. Les 99 % requis sont conservés. À titre diagnostique, la
palette naïve appliquée au pool seul ne donne la bonne boîte joueur que sur
1 111 des 2 048 fenêtres.

Les cinq tests propres à cet adaptateur couvrent identité et permutation des
images, couleurs hybrides, support disjoint, occlusion blanche, clipping et
capture RAM avant/après un unique pas. La validation réservée doit suivre le
scellement et la revue de ce candidat ; aucun gate ni apprentissage n'est lancé.
