# Worker RGB dans un interpréteur explicite

Prototype du 9 septembre 2026. `serve_ordinary_rgb_v70.py` possède au maximum un
moteur `_OrdinaryRGBTraceEngine`. Il ne crée aucun descendant multiprocessing.
Un futur parent pourra donc terminer un worker bloqué sans abandonner une seconde
couche de processus. Le moteur et les règles de paiement existants sont conservés.

Le worker envoie d'abord l'identité de son runtime. Il n'ouvre un jeu qu'après
une configuration complète explicitement reçue. Les jeux admis sont ceux du
registre de développement déjà relu, en mode laboratoire et répétition4 ; les
quatre jeux réservés sont refusés avant construction.

Les commandes sont limitées à `init`, `observe_pair`, `step`,
`branch_pair_trace`, `report` et `close`. Leur numéro doit croître exactement de
un. Un moteur n'est construit qu'une seule fois, même si sa construction échoue.
Une erreur d'action arrête les commandes payantes ; les rapports restent lisibles.
Si aucun compte complet n'est disponible, le rapport le dit explicitement au
lieu d'affirmer un coût total nul. Une erreur sans description native complète
rend également la comptabilité incertaine.

Le canal binaire utilise un duplicata de stdout ; les sorties humaines et natives
vers stdout sont redirigées vers stderr avant l'inspection d'ALE. Une erreur
pendant la sérialisation ou l'écriture ferme le serveur : aucune seconde réponse
n'est ajoutée derrière une première réponse partielle. Le parent devra garder
les coûts connus et ne jamais répéter une commande dont la réponse est perdue.

Les tests utilisent des moteurs synthétiques. Un seul test de sous-processus
envoie uniquement `close` après le message d'accueil : inspection de la version
et du binaire ALE, aucune configuration de jeu, ROM ou instance ALE créée.
Les défaillances initiales des tests venaient du compte de leur fixture, dépourvu
des champs natifs de démarrage ; sa méthode de compte synthétique est maintenant
explicite, sans modifier le moteur ni les anciennes fixtures.

Ce prototype attend sa revue. Le client avec délai borné, drainage de tous les
workers, arrêt global, conservation des comptes et validation du runtime reste
à construire. Aucun raccord natif ni inférence en ligne n'est encore autorisé
par les seuls tests de ce serveur.
