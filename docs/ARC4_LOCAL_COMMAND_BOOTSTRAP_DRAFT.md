# Amorçage des commandes : hypothèse locale distincte des effets sémantiques

Note prospective du 10 septembre 2026, issue d'une revue contradictoire de
`ARC4_VISION.md` §2.1–2.2 et `ARC4_BOOTSTRAP_LIMITS.md`. Aucun adaptateur n'est
implémenté ou entraîné, aucun protocole d'essai supplémentaire n'est gelé ici.

Avec le même passé et le même effet UNKNOWN, le parser déterministe ne peut pas
produire deux prédictions différentes selon une commande qu'il ne reçoit pas.
Le mouvement déjà visible dans le passé ne supprime pas cette ambiguïté.
L'architecture actuelle attend ME pour découvrir les effets, alors que P/C
utilisent déjà une hypothèse d'effet pour apprendre les représentations de ME.

Un prochain candidat pourrait distinguer un **code exploratoire local de commande**
des **douze effets sémantiques transférables**. Dans le babillage, une table propre
au contrôleur transformerait la commande émise en un code appris avec P. Le code
disponible pour prédire dépendrait seulement de cette commande connue. Les
images futures seraient uniquement des cibles observées après l'action ; ni leur
mouvement vrai ni la réalisation cachée des boutons collants n'entreraient dans
le code. Le parser garderait les images passées pour interpréter la commande
dans son contexte.

Cela modifierait explicitement l'ordre de l'amorçage prévu par la vision :
conditionnement local avant ME, puis éventuel alignement sémantique depuis les
propres observations et expériences. Un code de dimension douze n'est pas par
définition le vecteur des douze effets déjà définis. Cet alignement resterait
inconnu tant qu'il n'est pas établi.

La politique pourrait conserver son interface sémantique sans voir les boutons
si ce raccord reste dans le babillage et la perception. Lui faire choisir les
codes locaux non alignés reviendrait à changer son espace d'action ; ce serait
un écart à déclarer, pas une simple implémentation de l'interface actuelle.

Une reconstruction conditionnée correcte ne rendrait pas les codes identifiables.
Permutations, transformations compensées par le parser, commandes équivalentes,
réponses occultées et mémorisation resteraient possibles. Le changement visible
pourrait aussi concerner un objet indirectement affecté. Ce candidat retirerait
une ambiguïté d'entrée, sans prouver ME, un vocabulaire sémantique ou son transfert.

La priorité immédiate reste la perte du signal visuel : le fit à priorités
`craft_v6_out/v70_priority_motion_fit/fit_v0.json` s'est terminé avec une erreur
supérieure au seuil, et des reconstructions presque noires. Ajouter l'identité de la
commande ne corrigerait pas à lui seul cet échec, qui apparaît déjà avec deux
passés distincts. Aucun nouvel ajustement conjoint n'est lancé dans cet état.

Après traitement explicite de ce problème P, un seul test pourrait isoler la
question du conditionnement : comparer un code local appris à une commande
rendue constante, avec exactement le même passé suivi de deux commandes et de
conséquences différentes. Un second passé serait réservé aux mesures dès le
départ. Seuls les futurs RGB superviseraient l'apprentissage. Budget, permutations,
seuils et arrêt seraient déclarés avant l'essai. Un succès mesurerait une
réutilisation du conditionnement sur ce jouet ; un effondrement commun resterait
non concluant sur l'adaptateur. Aucun critère du gate 0 n'est modifié par cette note.
