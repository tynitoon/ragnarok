# Localiser le coût de la chaîne native avant une autre optimisation

Proposition du 10 septembre 2026, après le benchmark rectangle `18bbeb6`.
Aucun profil supplémentaire n'est lancé par cette note. La collecte descriptive
des effets d'actions reste la prochaine exécution ARC4 prévue.

Le rectangle conserve les seize rapports de la référence exactement, mais son
débit mesuré est de 518,347 pas/s contre 580,79 pour la référence historique.
La charge ARC3 n'était pas identique : ces mesures ne permettent pas d'attribuer
une régression au changement, ni de démontrer un gain. Les seuils proposés de
1 000/2 000 pas/s restent inchangés. Les versions cache et gather n'ont pas non
plus établi le gain recherché. Le reçu indépendant
`v70_mspacman_rectangle_cpu/native_chain_review_v0.json` (SHA256 commençant
par `0a1dda44`) confirme les preuves, les 171 364 charges et ces limites.

Un profil existe déjà : `v70_mspacman_cpu/offline_scoring_profile_v0.json`.
Il a mesuré 8 192 images archivées avec une cadence synthétique et cProfile :
13,15 s dans le décodage, 7,32 s dans le regroupement, 4,44 s dans l'historique
des mouvements, 3,14 s dans la comparaison et 1,53 s dans la référence RGB.
Ces durées instrumentées ne sont ni additives à celles du benchmark natif,
ni une mesure de son débit. Elles excluent l'émulation, les branches, les
restaurations et l'IPC, et portent sur une autre trajectoire.

La question utile suivante est donc : sur une trajectoire native réelle déjà
connue, le temps se concentre-t-il encore dans le décodage et le regroupement,
ou dans des opérations absentes du profil hors ligne ?

Une expérience possible, à préciser dans un protocole exécutable avant tout
lancement, est un profil de la référence native inchangée sur le seul worker0
historique : Ms. Pac-Man, seed18000,2 048 décisions, commande `(step//4)%18`,
repeat4, sticky0,25, horizon et branches toutes128 décisions inchangés. Le
plafond proposé est de30 000 frames facturées et300 secondes coopératives.
L'absence des quinze autres workers et du transport devra être explicite :
ce profil ne répondrait pas à la question d'un coût d'IPC ou de contention.
Il sert à identifier les fonctions coûteuses, sans produire de nouveau débit
comparatif ni remplacer la baseline existante.

Il faudrait enregistrer séparément construction, boucle et clôture, les coûts
complets du moteur et les comptes de cProfile. Les temps propres de fonctions
peuvent être regroupés ; les temps cumulés ne doivent pas être additionnés,
car ils incluent leurs sous-appels. Les rapports déterministes pertinents
devront égaler ceux du worker0 historique ; la seule différence prospective
de plafond comptable devra être explicitement exclue de cette égalité. Aucune
normalisation après observation d'une divergence n'est admise.

Avant toute construction : sources et protocole committés, revue indépendante,
vérification du runtime et du build natif, puis marqueur de tentative durable
et exclusif. Les coûts partiels ou inconnus et toute erreur seront conservés.
Les captures, entraînements et autres profils CPU seront arrêtés pendant la
mesure. Un échec ne déclenche pas une répétition ni un autre seed.

La construction de ce profil est différée jusqu'au résultat de la collecte des
effets d'actions. Sa valeur devra alors être comparée à celle d'un diagnostic
du transport : multiplier les versions optimisées sans identifier le coût
dominant ne résout pas le préalable de performance. Aucun GPU, apprentissage,
jeu réservé, nouvelle calibration ou résultat de transfert n'est autorisé par
ce brouillon.
