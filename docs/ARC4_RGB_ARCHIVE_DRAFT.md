# ARC 4 — archive privée, prototype à revoir

L'archive conserve une situation pendant que le jeu ordinaire continue. Les
anciens jetons `clone` ouvraient une session où toutes les avances suivantes
étaient expérimentales ; ils ne convenaient pas à cette fonction.

`rgb_archive_v70.py` propose une API séparée : enregistrer un état, lancer une
paire de branches depuis cet état, supprimer explicitement un jeton. Le moteur
conserve les octets natifs, les deux images du pooling et le RNG/historique
sticky. Il ne rend qu'un identifiant aléatoire local au worker. Aucun chargement
de fichier ou import d'état externe n'est prévu.

Une paire depuis une archive coûte au maximum **196 charges pour deux branches
de huit frames** : charger l'archive, la recharger avant le contrôle, puis revenir
au point courant, soit trois restaurations de60 et seize frames. La paire au
point courant garde son coût de136. L'admission réserve le coût complet avant
la première restauration ; les branches tronquées ne créent pas d'images.
Toute frame exécutée et chaque restauration restent facturées en cas d'erreur.
Une erreur interne ferme la trajectoire ; le registre reste consultable.

Le plafond porte sur20000entrées et2milliards d'octets sérialisés, sans prétendre
borner exactement la mémoire du processus Python. Les buffers temporaires de
capture/décodage, les traces et les index Python s'y ajoutent. Il n'y a pas
d'éviction automatique : la future politique de cellules devra choisir
explicitement les entrées à remplacer. Ces paramètres et cette comptabilité
mémoire doivent être revus avant un protocole de collecte.

Ce prototype sert aux expériences rétrospectives avec retour au jeu courant.
Le retour durable à une frontière (E4), le classement des cellules, le transport
fermé et une validation native prospective restent à réaliser. Aucun des
protocoles déjà gelés, ni leurs sources, n'est modifié. Aucune graine nouvelle
ni entraînement n'est autorisé par ce document.
