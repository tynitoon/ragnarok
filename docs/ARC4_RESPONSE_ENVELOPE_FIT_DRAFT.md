# ARC 4 — un fit auxiliaire sur les douze ancres de développement

Choix prospectifs du 10 septembre 2026, avant tout ajustement sur des images
Atari. Ce protocole ne modifie aucun ancien fit, seuil ou STOP et ne constitue
pas le gate 0. Aucun jeu réservé ne sera ouvert.

La question est limitée : les deux images passées permettent-elles à un petit
réseau de mieux ajuster les contrastes observés qu’une entrée constante ? Les
douze ancres ont déjà servi à concevoir cette tâche. Même un résultat positif
resterait compatible avec une mémorisation de leurs apparences locales.

## Réseau et témoins fixes

Les deux RGB natifs sont concaténés en six canaux, dans leur ordre temporel,
et divisés par 255. Cinq convolutions 3×3 de 16 canaux, de dilatations 1/2/4/8/16,
avec SiLU et padding par réplication, précèdent une sortie 1×1 à un canal.
Il n’existe aucun identifiant de jeu, de commande ou d’ancre, aucune coordonnée,
normalisation de mini-lot, attention ou couche dimensionnée par la résolution.
Les cibles futures n’entrent jamais dans le réseau.

Deux bras partent des mêmes poids frais, graine 764 : images réelles et entrée
entièrement nulle. Grâce au padding répliqué, le second ne reçoit pas un code
de position par les bords. Les prédictions initiales sur les vraies images et
la sortie zéro sont les deux autres témoins. Aucun poids antérieur n’est repris.

## Ajustement et mesure

Adam, taux 0,0003, paramètres usuels explicités dans le protocole machine,
sans décroissance ; norme de gradient limitée à 1. Les bras alternent dans
l’ordre visuel puis constant. Chacun effectue exactement 256 mises à jour.
Chaque mise à jour utilise les douze ancres entières dans l’ordre du manifeste.
Il n’y a ni sélection de pixels positifs ni omission des deux cibles vides.

La perte moyenne donne le même poids à chaque ancre. Lorsqu’elle contient les
deux classes : moitié de la BCE moyenne des positifs, moitié de celle des
négatifs. Pour une ancre vide, BCE moyenne de tous ses pixels. Aucun autre
objectif n’est optimisé et aucun poids de perte n’est ajusté après observation.

Pour chaque ancre non vide : IoU au seuil fixé de 0,5 et aire précision-rappel
non interpolée (average precision, scores ex aequo groupés). Moyenne par jeu,
puis entre jeux. Pour chacune des ancres vides : fraction de faux positifs,
séparée de l’IoU. Toutes les mesures individuelles et tous les témoins sont
conservés ; seule l’évaluation finale détermine le résultat, sans choix du
meilleur checkpoint.

Critère de faisabilité : les deux bras terminent 256 mises à jour, l’IoU moyen
visuel vaut au moins 0,25, son avance sur le meilleur des trois témoins vaut
au moins 0,10, et les faux positifs restent au plus à 0,1 % sur CHACUNE des
deux cibles vides. L’average precision est publiée, sans seuil additionnel.
Un échec reste un échec de ce candidat dans ce budget, pas une impossibilité
d’apprendre la perception ni une permission de changer les critères.

## Budget et arrêt

CPU un thread, CUDA masqué, aucun émulateur. Au maximum 512 mises à jour,
1 200 secondes de travail coopératif, évaluations et sérialisation des sorties
comprises, hors import du runtime et contrôles de provenance préalables.
Le dimensionnement synthétique effectué avant le fit mesure un unique
forward/backward du lot natif en 1,82 s, sans optimizer.step : 512 tels calculs
représentent environ 932 s avant les évaluations et les écritures. C’est la
raison du plafond de 1 200 s fixé ici ; la suggestion initiale de 900 s n’était
pas un protocole exécuté. Cette estimation ne garantit pas la durée réelle.

Tentative et journal exclusifs, poids initiaux/finals et prédictions conservés.
Arrêt sur plafond, erreur ou valeur non finie ; aucun second départ, recherche
d’hyperparamètres, nouvelle graine ou prolongation après le résultat. Si le
temps manque avant les deux bras et leur mesure finale, le résultat est
incomplet. Les mises à jour réellement effectuées restent comptées.

La construction des cibles, son résultat et sa revue indépendante doivent être
committés avant le préflight de ce fit. Celui-ci doit aussi être revu et committé
avant la tentative. Une réussite restera une faisabilité sur développement,
sans preuve d’objets physiques, de ME, de jeu appris, de généralisation ou de
transfert. Une validation future devra porter sur de nouvelles situations
choisies prospectivement, dans un autre protocole déclaré avant leurs pixels.
