# ARC 4 — premier réseau de perception, sans entraînement Atari

Le prototype `ragnarok/learning/parser_v70.py` reçoit uniquement une image RGB
poolée et un vecteur d'effets de douze valeurs. Le même réseau convolutionnel
32/64/96 est appliqué à l'échelle native et à la demi-échelle ; les caractéristiques
sont fusionnées avant les têtes d'embedding32, premier plan, déplacement,
contrôlabilité et solidité. GroupNorm et SiLU sont utilisés. Aucune couche
dépend de la résolution, du nombre de catégories ou du nombre de boutons.
Le nombre exact de paramètres doit être mesuré ; l'estimation1,5M du plan ne
constitue pas une mesure de cette implémentation à poids partagés.

Seule la tête de déplacement reçoit le vecteur d'effets. Le vecteur nul signifie
effet inconnu. Le passage des boutons du babillage à ce vocabulaire est encore
à résoudre : aucun bouton, sens de déplacement RAM ou contrôlabilité oracle
n'est injecté pour le fabriquer. Les caractéristiques visuelles sont calculées
indépendamment de l'effet soumis pour cette prédiction.

`pixel_prediction_v70.py` définit le déplacement comme le mouvement **en avant**
d'un pixel source vers l'image suivante. Il distribue bilinéairement les couleurs
aux destinations, moyenne les collisions et laisse les trous à zéro. La perteL1
porte sur l'image entière, trous compris. Elle ne doit pas être interprétée comme
un modèle complet de désocclusion ou de changements d'apparence.

Une cible de premier plan compare, au pixel source, l'erreur prédite en restant
immobile et celle en suivant le déplacement prédit par le réseau. L'amélioration
relative positive devient la cible, seulement pour une norme de déplacement
supérieure à0,25pixel et une destination visible. Cette cible est détachée du
gradient ; un mauvais mouvement prédit peut donc produire une mauvaise cible.
Elle n'utilise ni fond classique, ni flot optique fourni, ni annotations d'objets.
Les coefficients, la cadence et les données d'apprentissage restent à figer.

Le test sur des translations synthétiques vérifie les coordonnées et le sens du
gradient ; il ne démontre aucune capacité à voir ou jouer. Les têtesC/S sont
présentes mais aucun objectifC/G/I/S, tracker, catégorie, ME ou lecteur n'est
encore branché. Il n'y a ni poids préentraînés, ni lecture des captures
d'instrumentation, ni lanceur d'apprentissage dans ces fichiers.

Vérification CPU explicite avec `C:/Python314/python.exe`, PyTorch2.11.0+cu126,
un seul thread de calcul, tenseursCPU. La confirmation ARC3 conserve le GPU.
Aucun gate0 ni entraînement Atari n'est autorisé par ce prototype.
