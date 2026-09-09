# Une limite constructive de la reconstruction et du premier plan

Témoin externe du 10 septembre 2026. Aucun modèle appris, aucune optimisation,
aucune archive réelle et aucun émulateur ne sont utilisés. Les anciennes sources,
mesures, critères et budgets restent inchangés. Les étiquettes physiques ci-dessous
définissent uniquement une scène de test du juge ; elles ne supervisent aucun agent.

Le décor déclaré est un fond noir immobile, recouvert par un rectangle blanc
opaque de deux lignes et trois colonnes. Le rectangle se déplace horizontalement
de quatre pixels. Les images correspondent à la source et aux futurs du petit
diagnostic précédent, mais cette scène précise explicitement leur interprétation
physique ; les images seules ne suffiraient pas à imposer cette interprétation.

Avec le déplacement correct des seuls pixels blancs et un fond immobile,
`forward_warp_rgb` donne une couverture de deux à chaque pixel de destination :
un pixel blanc arrive sur un pixel noir. La normalisation de la collision fait
leur moyenne et produit du gris 0,5, alors que l'objet opaque déclaré doit être
blanc. La perte native est exactement **0,01171875**, et non zéro. Elle ne passe
pas le seuil strict du diagnostic précédent. Le fond nouvellement découvert à
l'ancienne position est noir dans ce témoin ; les trous noirs du reconstructeur
y sont donc compatibles avec le décor.

Un autre champ, physiquement faux dans cette scène, produit pourtant exactement
le futur : il déplace aussi les six pixels noirs de destination de quatre pixels
vers le bas, à un endroit noir. Ils ne diluent plus le rectangle blanc arrivant.
La perte d'image devient zéro. Mais `motion_foreground_target` attribue alors
une cible supérieure à 0,99 à ces six pixels sources noirs, en plus des six
pixels blancs. Leur déplacement artificiel explique leur couleur mieux que
l'immobilité, car l'objet blanc recouvre leur position dans le futur.

Ces constructions fonctionnent pour les déplacements de quatre pixels vers la
gauche et vers la droite. Elles démontrent une incompatibilité de la moyenne
des collisions avec **cette scène à objets opaques et fond statique**, ainsi
qu'une ambiguïté supplémentaire de la cible auto-générée de premier plan.
Elles ne démontrent pas que les réseaux précédents ont adopté ces champs, ni
que tous les échecs d'ajustement ont cette cause, ni la fréquence du problème
dans Atari. Elles ne changent pas rétroactivement le verdict de ces essais.

La borne ne dépend même pas du déplacement précis des pixels blancs, tant que
le fond reste immobile. À chacune des six destinations blanches, le fond apporte
une couverture de 1. Si les pixels blancs y apportent un poids total `w_i`,
la luminosité prédite vaut `w_i / (1 + w_i)`. Les six pixels sources blancs
apportent au plus six unités de poids au total, donc `sum(w_i) <= 6`.
Par concavité de `w / (1 + w)`, la somme des six luminosités vaut au plus 3.
L'erreur native sur ces seules destinations vaut donc au moins
`(6 - 3) / 256 = 0,01171875` ; les erreurs ailleurs ne peuvent que l'augmenter.
C'est une borne mathématique pour la scène et le reconstructeur déclarés,
pas une nouvelle mesure sur les champs optimisés. Le fond noir sans texture
reste impossible à suivre physiquement depuis ces seules images.

Conséquence de conception à examiner : traiter explicitement les recouvrements
et la visibilité avant d'interpréter tout déplacement expliquant la couleur
comme du premier plan. Un simple succès de reconstruction ne pourrait pas
valider cette interprétation. Ce document ne choisit pas encore de nouveau
reconstructeur, de cible, de régularisation ou de protocole d'entraînement.

Sources : `ragnarok/learning/pixel_prediction_v70.py` (inchangé) et
`tests/test_prediction_occlusion_limits_v70.py` ; les exemples sont construits
indépendamment des champs optimisés conservés dans le résultat du témoin libre.
