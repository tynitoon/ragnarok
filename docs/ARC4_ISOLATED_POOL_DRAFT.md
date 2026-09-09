# Client RGB avec interpréteur enfant explicite

Prototype du 9 septembre 2026. `isolated_rgb_pool_v70.py` permet au processus du
réseau de sélectionner l'exécutable Python de chaque worker sans modifier les
environnements installés. L'identité attendue du runtime doit être fournie par
un futur protocole prospectif ; une identité différente empêche toute commande
d'initialisation, pour tous les workers. Aucun jeu réservé n'est accepté.

Le client possède1..16 processus directs, avec trois threads de transport chacun
pour lire, écrire et drainer stderr. La taille des requêtes est limitée à4096
octets ; les réponses suivent les limites du codec RGB. Les files sont bornées,
les logs gardent au plus16blocs de4096octets. Le délai d'un lot est partagé entre
ses réponses, au lieu d'accorder un nouveau délai complet à chaque worker.
La récupération des rapports après échec a son propre délai partagé. Ces bornes
ne sont ni une limite de RSS total, ni une garantie d'arrêt du système hôte.

Toutes les commandes envoyées sont drainées ou déclarées perdues avant de rendre
une erreur. Les rapports sont gardés dès leur réception, avant les validations
ou copies supplémentaires. Une erreur contient les rapports rafraîchis disponibles.
Un transport perdu rend son dernier compte explicitement incomplet et termine
le processus concerné. Le pool entier refuse ensuite de nouvelles actions ;
seuls les rapports et la fermeture restent possibles. Aucun renvoi d'une commande
incertaine n'est permis. La fermeture arrête aussi les threads sans les laisser
bloqués à publier dans une file pleine.

Les observations sont les propres paires RGB ; chaque pas valide les traces
ordinaires existantes. Une expérience vérifie ses deux branches et retrouve le
même ancrage RGB. Les comptes, l'identité du runtime et les erreurs restent hors
des entrées du modèle. Le collecteur de données propres existant peut posséder
ce client, ce que couvre le test de raccord synthétique.

Huit tests passent sur le Python système. Sept utilisent exclusivement des peers
et moteurs synthétiques : trois comptes sont conservés après une erreur payée,
un quatrième coût possible n'est pas inventé après une réponse perdue, et aucune
initialisation ne passe une identité de runtime incorrecte. Le huitième lance
explicitement le Python ARC4 pour accueil puis fermeture uniquement : aucune
ROM ni instance ALE ; son processus et ses trois threads sont ensuite arrêtés.

Les sources attendent une revue indépendante. La future validation native devra
figer le runtime attendu, les configurations, commandes et budgets, ainsi que la
preuve de fidélité avec l'ancien transport, avant la première initialisation.
Le client ne définit aucun protocole d'apprentissage ou gain de vitesse. Un
appelant doit toujours fermer le pool dans un `finally` ; un arrêt brutal du
processus parent par le système ne dispose pas ici d'une garantie supplémentaire
de terminaison des enfants.
