# Seaquest : calibration des deux images sources

Le juge brut V5 est scellé et a passé le couple réservé17312/17362. Sa preuve
committée est exigée avant toute nouvelle construction d'émulateur temporelle.
Le candidat temporel est un module distinct, sans changement aux25sources brutes.

`ARC4_SEAQUEST_POOLED_PROTOCOL.json` fixe17321/17322 et8192décisions répétées
jusqu'à quatre frames chacune. Le contrôleur de collecte RGBV4 reste identique ;
il choisit une commande depuis l'image brute courante à chaque décision. Les
fenêtres enregistrent les deux dernières images et RAM réellement produites,
ainsi que le maximum exact renvoyé par le wrapper. Les resets et répétitions
tronquées restent comptés : frames réelles +128par construction, plafond1081472.

La définition utilise l'union spatiale des enveloppes par catégorie, connexité4.
Un contact par un côté joint les composantes ; un contact de coin seul les garde
distinctes. Le sous-type surface, `submarine` dont le sommet est y<57, est mesuré
séparément sur les deux côtés. Minimum50appariements, présence totale et sur
fenêtres positives99 %, géométrie à un pixel près99 %, sans succès vide.

Le checker RGB conserve la contribution d'une source si un canal gagne dans le
maximum ou si les trois canaux sont égaux. Cette attribution à deux sources
n'affirme pas que leurs identités sont récupérables depuis la seule image maxRGB.
Le juge RAM par enveloppes nominales reste à confronter aux suppressions de pixels.

Revue de conception Ohm : connexité, sous-type, appariement et bornes temporelles
explicités avant capture. Quatre tests du module et deux tests du captureur
passent, dont capture simulée→vrai lecteur à travers une reprise froide avec
répétition courte (5frames réelles +256de chargements=261).

Le code, le protocole et les prérequis ont été committés `2b03bd9` avant17321.
Développement terminé :8192décisions,32759frames ordinaires réelles et512frames
de quatre constructions, soit33271facturées. Capture55,54s,64archives conservées.
La baseline passe toutes les catégories et le sous-type surface. Tous les
appariements ont leurs bords à un pixel près : joueur7937, requin12083,
sous-marin8096, plongeur2685, missile ennemi3430, missile joueur6850 ; le
sous-type surface compte1029appariements. Présence positive minimale99,9622 %
pour le joueur ;1027boîtes de surface sont exactes. Les31empreintes sont conservées.

L'exécutable réservé distinct et ses33sources ont été revus, puis committés
`e911d36`. Le sceau complet a été committé `f62333f` avant ouverture de17322.
Le marqueur de tentative précède même le constructeur ; une erreur ne permet
pas de recommencer discrètement. Onze tests orchestration/guard Git passent.

Réservé17322 PASS :8192décisions,32756frames ordinaires+640froides=33396facturées,
cinq constructions. Joueur7879/7882boîtes within1 (99,9619 %), missiles joueur
6566/6571 (99,9239 %), toutes les autres boîtes appariées within1. Le sous-type
surface est couvert :159appariements exacts. Présence positive minimale99,9493 %
pour le joueur. Les huit erreurs de géométrie restent dans le rapport, sans
correction après ouverture. Tous les critères annoncés sont inchangés.

Les33sources sont maintenant scellées, y compris captureur, calibrateur et
exécutable réservé. Ajouter des adaptateurs distincts pour le débit. Ce résultat
porte sur les instruments, pas sur un apprentissage par l'agent.
