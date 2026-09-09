# Raccord prospectif entre les deux environnements Python

9 septembre 2026. L'interpréteur système contient PyTorch ; l'environnement ARC 4
contient ALE et les dépendances figées des instruments. L'inférence sur archives
permet de tester le modèle sans modifier ces environnements. Pour la collecte en
ligne, il faudra relier explicitement les deux interpréteurs. Un worker démarré
implicitement depuis le processus PyTorch ne garantit pas le runtime ALE validé.

`rgb_wire_v70.py` prépare uniquement le format de transport : en-tête JSON typé
et octets RGB uint8, sans pickle ni reconstruction de classe choisie par un
message. Les deux seules classes de traces reconstruites sont celles déjà
définies pour les observations ordinaires et les branches. Les métadonnées sont
des valeurs primitives, listes, tuples ou dictionnaires ; la comptabilité reste
externe au modèle.

Le format conserve les préfixes réellement livrés, y compris zéro image. Chaque
tableau reçu a son propre stockage. Les limites sont un en-tête de1Mio,64Mio de
pixels par message,8192nœuds et16niveaux ; elles bornent le message, pas le RSS
total ni le temps d'attente d'un pipe. Une erreur doit fermer la connexion, sans
resynchronisation ou répétition d'une commande qui aurait pu être payée.

Le codec vérifie la structure des images, pas leur provenance ni la récurrence
du pooling : celles-ci restent la responsabilité du serveur fermé, du collecteur
et des validateurs existants. Une image fabriquée de la bonne forme ne devient
pas une observation propre grâce au transport.

À construire et revoir : serveur enfant lancé avec l'interpréteur explicite,
liste fermée de commandes, séparation des logs et du flux binaire, délai borné,
arrêt et conservation des coûts après réponse manquante. Aucun serveur, émulateur,
nouveau protocole natif ou changement d'environnement n'est lancé par ce codec.
