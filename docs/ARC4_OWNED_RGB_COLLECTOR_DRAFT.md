# ARC 4 — propriété des opérations et expériences RGB

Prototype sans lancement natif ni apprenant. Un collecteur possède exclusivement
son pool : aucune autre partie du programme ne doit y envoyer des commandes.
Cette règle d'architecture n'est pas une protection contre un appelant qui
modifierait volontairement les attributs privés Python.

Chaque opération reçoit un identifiant propre, croissant dans une session.
Chaque worker possède un ancrage courant lié à cette session, une époque et
une horloge calculées uniquement par la chronologie RGB. Une action ordinaire
remplace l'ancrage ; une paire de branches restaurée le conserve. Une demande
de branche portant un ancrage ancien ou étranger est refusée avant toute dépense.
Le collecteur vérifie aussi les buffers de l'ancrage avant et après la paire.
L'égalité des pixels ne prouve pas l'égalité d'un état caché : la garantie repose
sur cette propriété exclusive des commandes et sur le moteur de replay relu.

Les deux branches viennent obligatoirement d'un seul appel de transport. Leurs
préfixes réels, leur pooling depuis l'ancrage et leurs arrivées sont vérifiés.
Cette première API fixe le contrôle à NOOP. Le bouton demandé reste un journal
de la propre commande ; le modèle doit recevoir seulement l'hypothèse d'effet
à douze composantes, copiée avant la commande. Une mauvaise correspondance
bouton→effet reste une erreur possible de l'agent, pas une vérité fournie.

Une opération réussie peut être retirée une fois comme nouvelle expérience.
Ses images peuvent ensuite être réutilisées pour des mises à jour explicitement
prévues, sans les compter comme de nouvelles parties ou images émulées. Le
nombre d'opérations en attente est borné, à 64 par défaut ; une saturation est
refusée avant une commande, sans effacer d'anciennes expériences. Les données
retournées sont copiées, elles ne réécrivent pas les ancrages privés.

Une réponse payée incomplète, un mauvais préfixe ou une erreur tardive arrêtent
le collecteur. Les ancrages ne sont pas partiellement validés, mais les actions
déjà effectuées ne sont pas rembobinées. Les rapports et anciennes réussites
en attente restent accessibles. Le futur lanceur doit enregistrer la tentative
avant tout lancement de pool et préserver les coûts inconnus après perte de
transport. Le prototype ne résout pas encore cette persistance sur disque.

Les tests utilisent uniquement des moteurs synthétiques. Le raccord au parser,
les transitions sans progrès, l'unité temporelle des objectifs, le protocole
de collecte et la validation avec émulateurs réels restent à vérifier avant un
apprentissage. Aucun indicateur ALE de mort/reset n'est transmis aux expériences.
Les rapports comptables restent réservés au lanceur et à l'évaluation externe,
hors des entrées du parser ou de la politique. L'arrêt du collecteur est marqué
avant de construire le texte d'une erreur : même une panne mémoire pendant cette
journalisation ne doit pas permettre une nouvelle commande payante.
# Sérialisation des opérations publiques

Revue contradictoire : avant ce correctif, deux threads pouvaient exécuter deux
steps et rendre les réponses dans l'ordre inverse. L'ancrage final pouvait alors
revenir aux pixels antérieurs malgré huit images réellement exécutées, sans
arrêt du collecteur. Ce témoin est conservé comme défaut de la version b63831d.

Chaque méthode publique (`step`, `branch`, `take`, `current_anchor`, `report`,
`close`) porte désormais un verrou non bloquant et non réentrant. Tout appel
concurrent est refusé avant commande, copie ou mutation ; il ne modifie pas
l'opération déjà en cours. Le verrou couvre le paiement, la réponse du pool et
le commit du collecteur. Il est libéré même après exception. Les rapports et la
fermeture se font après la fin de l'opération, jamais au milieu de son commit.
La construction de l'objet doit toujours précéder son partage ; les accès
directs au pool restent interdits par le contrat de propriété exclusive.
