# ARC 4 — validation conditionnelle sur de nouvelles trajectoires

Choix prospectifs du 10 septembre 2026, pendant le premier fit, avant son
résultat final et avant toute nouvelle image de validation. Ce document ne
lance aucun calcul. Une seule validation est envisagée, uniquement après
clôture et revue indépendante d’un fit ayant atteint son critère de faisabilité.
Un fit négatif ou incomplet ne déclenche aucune collecte de ce protocole.

## Population et séparation

Les mêmes six jeux, dans le même ordre : Pong, Breakout, Freeway, Seaquest,
Ms. Pac-Man, Asteroids. Aucun jeu réservé. Nouvelles graines fixées ici :
19490 pour l’émulateur, 19491 pour Random, réinitialisées par jeu. Même marche
ordinaire à commandes aléatoires tenues quatre décisions ; deux ancres aux
seuils de 10 000 et 20 000 images ordinaires livrées ; commandes 0…17 contre
NOOP sur huit images futures. Aucune ancre de remplacement.

Les douze lignes sont réservées dès le début. Une paire ou un horizon manquant
rend l’enveloppe complète inconnue : l’union disponible est seulement une borne
inférieure, jamais une cible négative ou une cible complète. Un contrôle NOOP
incohérent invalide l’instrument. Ces cas et leurs coûts restent publiés.

Les poids finaux visuels et constants, les poids initiaux, le constructeur de
cibles et le scorer devront être scellés avant collecte. Aucune mise à jour,
adaptation, sélection de checkpoint ou normalisation recalculée. L’initial et
le visuel reçoivent les deux vrais RGB passés ; le constant reçoit zéro ; la
sortie zéro reste un témoin. Aucun autre champ n’entre dans les réseaux.

Les signatures des passés seront comparées à toutes celles du développement.
Chaque doublon sera signalé, conservé et jamais remplacé. Par prudence, une
validation contenant un tel doublon ne pourra pas obtenir le verdict global
de généralisation ci-dessous. Cela n’efface pas ses mesures descriptives.

## Critère avant données

Pour chaque ancre complète non vide : IoU au seuil 0,5 et average precision
avec scores ex aequo groupés. Moyenne entre ancres non vides d’un jeu, puis
moyenne égale entre jeux. Toutes les ancres vides ont leur fraction de faux
positifs publiée séparément. Conserver les dénominateurs complets, incomplets,
non vides et vides, et tous les témoins par ancre.

Couverture requise : douze ancres complètes, aucun passé dupliqué depuis le
développement, au moins cinq jeux avec une ancre non vide, et au moins deux
ancres vides. Si elle manque, verdict global NON CONCLUANT sans complément
de collecte. Si l’instrument est invalide, le résultat scientifique est invalide.

Avec cette couverture, réussite si l’IoU visuel moyen est au moins 0,25,
son avance sur le meilleur des trois témoins au moins 0,10, et les faux
positifs au plus 0,001 sur CHAQUE ancre vide. Sinon échec de cette validation.
AP est descriptive. Aucun seuil n’est ajusté après observation.

## Budgets et portée

Collecte : 30 000 images facturées par jeu, 180 000 au total, plafond commun
expériences/restaurations de 20 %, 900 secondes coopératives, 1 Go d’archives,
un worker à la fois. Construction dérivée : 300 s CPU, sans nouvelle émulation.
Évaluation des trois réseaux figés et des témoins : 180 s CPU, zéro update,
poids vérifiés avant/après. Imports et contrôles de provenance préalables sont
mesurés séparément. Le protocole machine et ses règles de clôture devront être
revus avant toute exécution ; aucun dépassement ou retry implicite.

Un PASS signifierait une généralisation de cette tâche de contraste à de
nouvelles trajectoires des jeux connus. Les deux ancres d’un jeu restent
corrélées ; ni les pixels ni les commandes ne sont des répétitions indépendantes.
Deux images peuvent ne pas identifier tous les futurs possibles, notamment
en présence d’état caché ou de boutons collants. Ce test ne prouve ni objets
physiques, ni ME/C, ni jeu maîtrisé, ni transfert vers un autre jeu, ni vitesse
d’apprentissage supérieure. Il ne remplace pas le gate 0.
