# Tête de priorité prédite depuis les observations disponibles

Candidate de modèle distincte, sans entraînement ni raccord à un collecteur.
`PriorityMotionParser` contient un `OrderedMotionParser` initialisé en premier,
avec exactement les mêmes poids vierges pour une même graine. Il ajoute une
tête recevant les deux écrans RGB ordonnés, l'embedding propre de 32 dimensions
et les deux composantes du déplacement propre. L'effet choisi n'intervient
dans la priorité qu'à travers ce déplacement prédit. Aucune image future,
priorité photométrique tirée du futur, identité, boîte ou masque n'est une entrée.

La nouvelle tête comporte deux convolutions 3×3 avec GroupNorm et SiLU, puis
une convolution 1×1. Sa sortie vaut `20 * tanh(logit / 20)`, donc reste dans
le domaine de l'opérateur de reconstruction à priorités. Les poids et biais
de sa dernière convolution commencent à zéro : la priorité initiale est
uniforme et les cinq sorties visuelles originales restent identiques.
Cela donne d'abord un gradient à la dernière couche de priorité ; ses couches
précédentes ne reçoivent pas encore de gradient utile à ce premier passage.

Ce modèle utilise une composition, sans hériter des anciens types de parser.
Sa sortie contient explicitement `visual` et `log_priority`. Les anciens
propriétaires ne doivent pas l'accepter comme substitut avant de payer une
commande : un futur propriétaire devra vérifier ce nouveau contrat. Aucun
ancien flux ou propriétaire n'est modifié par cette candidate.

Les tests vérifient l'identité des poids et sorties initiales, le chemin de
gradient de la seule perte RGB vers la priorité et le mouvement, les bornes,
les données invalides et l'absence d'argument futur. Aucune mise à jour d'un
optimiseur n'est exécutée. Une borne de signature n'atteste pas la provenance
des tenseurs ; le futur raccord de données propres reste nécessaire.

Ce raccord ne démontre pas l'apprentissage de l'ordre d'occlusion. La priorité
peut encore masquer un mauvais mouvement ; son échelle n'est pas une probabilité
de visibilité. Les cibles de premier plan et de contrôle doivent encore traiter
les correspondances occultées comme inconnues. La candidate ne réemploie pas
l'ancien objectif complet P/m comme s'il était corrigé et n'introduit aucune
recette, nouveau fit, gate, collecte ou checkpoint Atari.
