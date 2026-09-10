# Essai unique d’ajustement du contrôle stationnaire

Déclaration du 10 septembre 2026, avant tout fit de ce contrôle. Condition
d’entrée : construction complète du contrôle stationnaire, certificat numérique
atteint et revue indépendante conforme. Sinon cet essai ne démarre pas.
Il reste distinct des cinq fits de mouvement et du fit Atari négatifs.

## Question et comparabilité

Le réseau exact possède une solution explicite pour cette famille. La question
est donc : son initialisation aléatoire et la procédure d’apprentissage déjà
utilisée atteignent-elles le critère de faisabilité dans le même budget de mises
à jour ? La solution du juge n’est jamais chargée ni enseignée à l’apprenant.

Conserver le réseau 10177 paramètres, la graine d’initialisation764, Adam3e-4
(betas0,9/0,999, eps1e-8, sans weight decay, amsgrad/foreach/fused faux),
clip de norme1, BCE équilibrée par ancre et par classe observée. Aucun changement
de normalisation, de profondeur, de loss ou de cadence par rapport au fit Atari.
Les deux bras démarrent aux mêmes poids ; aucun poids d’un essai antérieur.
Runtime, dtype float32 et ordre de construction sont conservés : aucune
consommation RNG entre le seed et le premier réseau, puis copie exacte pour
le témoin. Les paramètres et leurs signatures sont archivés.

Le processus ne charge ni fabrique ni poids ni logits du certificat. Il décode
uniquement les deux passés RGB et leurs enveloppes observées dans l’archive
du contrôle. Les futurs restent archivés mais ne sont pas décodés par le fit.
Les RGB, divisés par255, forment les six canaux ordonnés du seul input.
Les enveloppes issues des branches sont les seuls labels. Aucune couleur
codée en règle, coordonnée, identité de cas ou commande en entrée du réseau.
Les sources du juge sont vérifiées par hash comme provenance, sans import.

Les douze cas restent dans leur ordre et dans chaque lot. Les noms de six paires
servent seulement à conserver la pondération de mesure ; ce ne sont pas six jeux.
La famille contient dix cas non vides et deux vides. Le témoin à entrée nulle
a donc la même constante optimale analytique 5/12 que dans le fit Atari.

## Essai et critère inchangés

256 mises à jour visuelles et256 du témoin à entrée nulle, alternées dans cet
ordre à chaque étape. Évaluer uniquement l’initial visuel puis les deux bras
finaux. Le témoin de sortie nulle est calculé sans modèle. Publier toutes les
mesures, y compris celles d’un initial déjà performant, sans modifier la tâche.

Seuil0,5. IoU par cas non vide, puis moyenne au sein de chaque paire non vide,
puis moyenne des cinq paires non vides. Les vides n’ont pas d’IoU et sont évalués
séparément. L’AP groupe les scores ex aequo, sans recherche d’un autre seuil.

Le critère primaire exige ensemble :

- exécution complète des256 mises à jour de chaque bras ;
- IoU moyen visuel au moins0,25 ;
- avance d’au moins0,10 sur le meilleur des témoins entraîné constant,
  initial visuel et sortie nulle ;
- fraction de faux positifs au plus0,001 sur chacun des deux cas vides.

Ce seuil est volontairement celui de faisabilité déjà employé. Un succès ne
signifie pas ajustement parfait ou récupération de la solution connue. On
publiera aussi la BCE finale et son écart descriptif au certificat connu ; cet
écart ne participe pas au verdict et ne déclenche aucun essai supplémentaire.
La BCE du certificat est une valeur atteignable, pas une borne inférieure :
des logits de plus grande amplitude peuvent la réduire sur cette famille.
Trois cas non vides parfaits et sept entièrement manqués suffisent à une IoU
moyenne de0,30 : publier les douze lignes individuelles, notamment les cas
à1/2/4 pixels et aux coins, est donc indispensable. Les vides ont chacun
33600 pixels ; leur plafond admet au plus33 faux positifs.

Si le témoin initial réussit déjà, une clause d’avance manquée reste un FAIL
global, avec sa cause explicite. Aucun nouveau lot ni départ pour rendre le
témoin moins bon. Les passés identiques ne testent ni l’ordre temporel, ni
l’anticipation de déplacement, même en cas de succès.

## Coûts, clôture et STOP

Une tentative exclusive, CPU un thread, CUDA vide, au plus1200s de travail dont
10s réservées à la clôture,64MB de sorties binaires. Les216 pas synthétiques
de construction sont hérités, pas rejoués. Zéro nouvelle génération et zéro
nouvelle image émulée. Le nombre de forwards d’apprentissage et d’évaluation
sera compté séparément des mises à jour de l’optimiseur.

Les512 étapes sont un maximum fixe, sans arrêt anticipé au succès. Erreur,
valeur non finie ou dépassement : STOP, conservation des sorties et poids
partiels, compte explicite d’une éventuelle mise à jour incertaine, aucun retry.
Pas de changement de graine, lot, seuil ou budget après résultat.
Imports et preuves initiales sont exclus du temps de travail ; entrées décodées,
apprentissage, évaluations, sérialisation, fsync/hash, clôture du journal,
vérification finale de provenance et payload sont inclus. Seul le petit reçu
final et la console viennent après la mesure.

Sources, protocole, tests pertinents et préflight indépendant doivent être
committés avant la tentative. Les tests du lanceur utilisent des fixtures,
sans apprentissage du réseau réel ou lecture du lot natif. Revue du résultat
avant interprétation ou nouvelle décision technique.

## Conclusions permises

PASS : cette procédure s’ajuste assez pour le critère sur ce lot facile dans
ce budget. Elle n’est pas entièrement inopérante. Cela ne localise pas la cause
de l’échec Atari et ne réhabilite aucun essai antérieur.

FAIL complet : cette procédure n’atteint pas le critère dans ce budget alors
qu’une solution de l’architecture existe. Cela n’établit ni une impossibilité
d’apprentissage, ni un minimum local, ni l’effet causal d’un hyperparamètre.

INCOMPLET : aucune conclusion sur la faisabilité. Dans tous les cas : ni jeu
maîtrisé, ni généralisation indépendante, ni transfert, ni apprentissage
accéléré d’un jeu suivant démontrés. Les nouveaux fits Atari et le raccord
P→m→C/ME restent suspendus pendant cette calibration.
