# Validation conditionnelle de la famille stationnaire

Déclaration du 10 septembre 2026, avant lancement du fit stationnaire.
Ce document est prospectif ; aucun nouvel écran ni modèle n’est exécuté.

## Condition et question

Uniquement si le fit stationnaire se termine complètement, atteint son critère
global et passe la revue indépendante, préparer une unique évaluation gelée
sur de nouveaux écrans de la même famille. Un échec de l’une de ces conditions
laisse cette validation non exécutée, sans tirage de remplacement.

Le fit de développement peut satisfaire le critère en réussissant seulement
une partie des cas. Cette validation distingue une association qui fonctionne
sur d’autres tirages de la famille d’un succès limité au lot déjà appris.
Elle ne mesure ni un nouveau jeu ni la vitesse d’apprentissage ; les poids
restent gelés et aucune mise à jour n’est permise.

## Données fixées avant le fit

Même générateur stationnaire, dimensions210×160, horizon8, deux passés NOOP,
branches NOOP/commande1, cibles exclusivement par leurs différences observées.
PCG64, nouvelle graine775. Douze cas, mêmes nombres de pixels de réponse et
même ordre1/4/16/64/128/256/0/0/336/32/8/2 ; positions et nuisance RGB redessinées.
Les paires de mesure gardent leur seule fonction de pondération, sans entrée
dans le réseau. Chaque paire de RGB passée doit différer de toutes celles du
développement, par comparaison exacte des octets. Un doublon quelconque rend
la validation non concluante, sans nouveau seed ni remplacement de cas.

Douze écrans initiaux et216 pas synthétiques supplémentaires, comptés
séparément des216 pas de construction. Zéro image Atari et aucun jeu réservé.

## Mesure et limite

Les poids finaux visuel et constant ainsi que l’initial sauvegardé sont ceux
du fit revu ; aucun choix de checkpoint. Trois forwards d’évaluation au total,
un lot12 par modèle. Les deux modèles visuel/initial reçoivent seulement les
deux passés ordonnés ; le témoin constant reçoit leur tenseur mis à zéro.
Le témoin de sortie nulle est calculé sans forward. Aucun poids du certificat.

Conserver le seuil0,5, l’IoU moyenne par paire non vide puis entre les cinq
paires, l’AP groupant les ex aequo et les deux mesures de faux positifs vides.
Publier toutes les lignes individuelles. Le critère reste IoU visuelle≥0,25,
avance≥0,10 sur le meilleur des trois témoins et FP≤0,001 sur CHAQUE vide.
L’absence de mises à jour est obligatoire. Aucun balayage de seuil après mesure.

Une réussite établirait cette performance sur ces nouveaux tirages de la même
famille stationnaire. Elle ne démontrerait ni la loi exacte sur toute la famille,
ni une généralisation à une autre palette, ni le mouvement, le contrôle d’un
jeu, le transfert ou un apprentissage accéléré.

## Construction de l’instrument avant toute donnée

Un script et un protocole JSON distincts devront appliquer ces clauses,
être testés sur des fixtures et revus/committés avant la tentative. Une seule
tentative CPU1, CUDA vide,180s dont10s de clôture,64MB de sorties binaires.
Imports/preuves initiales hors temps ; génération, inférences, toutes les
sorties scientifiques, fsync/hash, journal et vérification finale inclus.
Seul le petit reçu final et la console après la mesure. Préserver les poids
et sorties lus ; aucun retry, ajustement, changement de famille ni extension.
Une erreur technique ou budgétaire rend l’exécution incomplète, sans PASS.
