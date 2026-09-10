# Calibration Atari : suffisance d’une recette linéaire locale

10 septembre2026, après les fits CNN Atari et stationnaires négatifs et la
calibration affine stationnaire positive. Une revue contradictoire de la
question a été effectuée avant cette déclaration. Ce document n’exécute
aucune expérience et ne modifie aucun protocole antérieur.

## Question et portée

Cette recette linéaire, utilisant les six intensités RGB locales des deux
images passées, atteint-elle le critère de développement fixé sur les douze
enveloppes observées Atari déjà archivées ? Il s’agit d’un nouveau candidat
choisi après observation de ce lot de développement. La déclaration
prospective de sa recette n’efface pas cette sélection adaptative.

Un PASS montrerait que ces indices locaux suffisent à atteindre ce critère
sur ce lot, éventuellement par des couleurs propres aux scènes et avec des
réussites inégales. Il ne prouverait pas que le contexte est inutile pour
reconstruire les enveloppes complètes. Un FAIL en16pas ne sépare pas une
optimisation incomplète, la limite d’un séparateur linéaire, des conflits entre
palettes ou les ambiguïtés de positions futures. Une baisse de BCE ne
certifie pas l’optimum ; l’optimum BCE ne garantirait pas le seuil IoU/FP.

## Données et séparation des rôles

Réutiliser seulement `v70_response_envelopes/targets_v0.npz`, ses passés RGB
et ses labels de différences de branches, dérivation et revue déjà scellées.
Douze ancres, dans l’ordre Pong/Breakout/Freeway/Seaquest/MsPacman/Asteroids,
deux par jeu,210×160, deux passés. Cibles de144/296/344/184/127/127/0/0/
109/184/78/54pixels. Les deux Seaquest vides restent dans toutes les mesures ;
ils décrivent les huit images observées, pas l’incontrôlabilité de Seaquest.

Zéro nouveau tirage, frame émulée, écran synthétique, RAM, modèle de référence
ou poids appris chargé. Les six intensités divisées par255 et le biais sont
les seules entrées. Les noms des jeux restent des métadonnées de pondération
et de publication. Les futurs ne sont fournis à l’apprenant qu’à travers les
labels déjà dérivés. Les cibles sont des coordonnées de réponses observées,
pas des identités d’objets ni un module C/ME.

## Recette unique

Reprendre le modèle affine partagé de sept paramètres float64, tous àzéro,
et Newton amorti7/4 déjà vérifié. Deux bras : RGB observés et six intensités
constantes nulles avec biais1. Chaque bras effectue16mises à jour, alternées
visuel puis constant à chaque étape. Aucun bras Adam supplémentaire, recherche
de taux, arrêt sur score, prolongation ou changement d’initialisation.

La BCE donne la même masse à chaque ancre et, dans une ancre mixte, la même
masse à chacune des deux classes ; une ancre pure conserve toute sa masse.
La borne Hessienne7/4 reste valable pour ces RGB bornés. L’amortissement ne
modifie que la résolution, sans pénalité dans l’objectif. Pas entier et garde
de descente : hausse supérieure à1e-12×(1+|perte précédente|) = arrêt technique,
avec conservation des coefficients et des compteurs partiels.

Évaluer l’initial commun et les deux bras finaux, plus la sortie nulle sans
produit de modèle. Objectif avant et après chaque mise à jour. Compteurs
attendus :67produits linéaires,67objectifs,32gradients,32Hessiennes,
32résolutions et32retours de mise à jour. Aucune autre évaluation.

## Résultat, budget et clôture

Critère inchangé : IoU moyenne non vide≥0,25, avance≥0,10 sur le meilleur des
trois témoins (constant appris, initial, sortie nulle), faux positifs≤0,001
sur CHACUNE des deux ancres vides. Seuil de probabilité0,5. Moyennes par ancre
puis par jeu non vide, AP avec ex aequo groupés. Publier toutes les douze lignes,
les deux trajectoires de BCE, logits, probabilités et coefficients initiaux,
finaux ou partiels. Aucun autre seuil recherché après les mesures.

Une tentative CPU1, CUDA vide,120s dont10s de clôture,64MB de sorties binaires.
Imports et preuves initiales précèdent le chronomètre ; depuis le décodage
jusqu’aux sorties scientifiques fsync/hachées, provenance finale et journal
fermé, tout est compté. Seul le petit reçu final et la console restent hors
temps. Écriture exclusive, aucune répétition ; une exécution incomplète
n’obtient pas de PASS scientifique. Sources et préflight indépendant exact
doivent être committés avant le lancement. Revue après scellement des sorties.

Les poids artificiels appris restent des résultats historiques et ne sont
pas des états initiaux. Comparer au CNN antérieur ne mesure pas le seul effet
du contexte : architecture, précision, initialisation et optimiseur diffèrent.
Cette tentative n’autorise ni les validations conditionnelles fermées, ni un
jeu réservé, ni le raccord P→m→C/ME, ni le gate0. Ce sera le dernier repère
linéaire local de cette série, sans enchaîner des réglages pour obtenir un PASS.

## Correction de provenance avant tout fit

Le préflight v0 s’est arrêté avant décodage, en5,4432969s : le premier résultat
Atari n’a pas le champ récent `final_provenance_verified`. Son schéma est
désormais vérifié explicitement par l’absence d’erreur et les deux contrôles
de provenance de sa revue historique épinglée. Les résultats plus récents
conservent l’obligation de leur indicateur final ; aucune valeur vraie par
défaut n’est admise. Trois fixtures ciblées couvrent cette distinction.

Le reçu négatif v0 et sa tentative restent scellés2a8d09e, SHA du reçu
5ab4cfe12c7e1d860227371dedb88bfd554af24e473aa6150fff4c368d0653ad.
La vérification corrigée utilisera un reçu v1 distinct lié à cet historique,
avec comptage cumulé du préflight sous60s. Aucun fit, tableau décodé, gradient
ou mise à jour n’a été consommé ; la recette scientifique reste identique.
