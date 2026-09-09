# Intégration temporelle du mouvement ordonné

Ce nouveau module utilise explicitement `OrderedMotionParser` et refuse l'ancien
parser dès sa construction. Les anciens modules, protocoles et résultats gardent
leurs sources. Il n'existe toujours aucun lancement de modèle sur un jeu réel
avec cette variante.

À chaque observation, les deux écrans RGB propres sont transmis dans leur ordre.
Les têtes f/m/c/s utilisent encore le maximum des images ; seul d reçoit en plus
les deux images séparées. Les groupes et le suivi utilisent les prédictions
propres du modèle. P vise toujours quatre images réellement jouées et I compare
des observations distantes d'exactement huit images ordinaires.

La mémoire I conserve les deux images uint8 et le suivi détaché, puis recalcule
l'embedding avec les poids courants. Elle ne garde ni graphe ni embedding périmé.
Cela double le stockage RGB de chaque entrée par rapport au flux initial ; la
fenêtre reste de huit images, donc au plus neuf horodatages. Le recalcul utilise
le parser complet, sans prétendre optimiser ce coût. Les mémoires restent
distinctes par flux et époque.

Les modifications du suivi et de la mémoire ne sont publiées qu'après le calcul
des objectifs. Les cinq têtes sont vérifiées finies avant de construire ces
cibles. L'appelant doit toujours attester la provenance et l'horloge ; cette
classe seule n'autorise ni collecte, ni optimiseur, ni validation native.

Les tests CPU synthétiques couvrent l'ordre des entrées, la mémoire sans alias,
le décalage exact de I, un échec tardif sans changement du suivi, et le passage
du gradient de P à la branche ordonnée sans modification des poids. Aucun de
ces tests ne démontre une direction apprise ou une stabilité d'entraînement.
Les futurs propriétaires qui exécutent des actions devront eux aussi vérifier
le contrat du modèle avant la première commande payée. L'héritage Python du
candidat ne le rend pas compatible avec l'ancienne signature pooled-only.

Les adaptateurs `OwnedOrderedPixelObjectives` et `OwnedOrderedPerceptionSession`
sont maintenant préparés dans un fichier séparé. Ils réutilisent sans modifier
les méthodes d'action, de verrouillage, d'arrêt et de conservation des données
des anciens propriétaires. Seules l'inférence P/C et la construction du flux
temporel passent par les entrées ordonnées. Le modèle exact, ses paramètres
float32 finis sur un seul appareil et les dimensions des ancrages sont vérifiés
avant les commandes de ces propriétaires. La construction initiale de l'émulateur
reste à la charge de l'appelant, ainsi que sa fermeture si le propriétaire ne
peut pas être construit. La mémoire temporelle conserve l'effet UNKNOWN tandis
que P/C utilise l'hypothèse d'effet enregistrée pour l'action correspondante.

Cinq tests supplémentaires passent sur moteur synthétique : rejet de l'ancien
modèle sans action, ordre/effet enregistrés, maintien de la mémoire pendant une
expérience, coûts exacts et conservation d'une action payée après tête non finie.
Le premier test d'expérience avait demandé le bouton5 à une fixture qui ne
possède que0/1 ; seul ce bouton de test a été corrigé. Aucun jeu réel n'a été
ouvert. Ces adaptateurs et le flux ordonné attendent leur revue indépendante.
