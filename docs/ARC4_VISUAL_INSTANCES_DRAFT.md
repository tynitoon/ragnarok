# ARC 4 — objets issus des prédictions du réseau

`visual_instances_v70.py` transforme les propres sorties du parser en instances.
Le premier plan est son logit positif ; une procédure4connexe trouve les
composantes, puis un k-means déterministe peut les séparer si leurs embeddings
appris varient. Il n'y a ni typeRAM, ni ROI de jeu, ni palette connue, ni masque
d'un extracteur classique. Aucun petit objet n'est supprimé par sa taille.

Les paramètres candidats sont une variance0,2, huit groupes par composante et
douze itérations k-means. Une variance encore élevée après la limite est
signalée. Le plafond de4096instances provoque une erreur explicite, jamais
une sélection silencieuse des grands objets. Ces limites sont des paramètres
du prototype à revoir avant gel, pas des seuils ajustés à des captures Atari.

Les composantes initiales sont connexes ; les groupes issus du k-means ne le
sont pas nécessairement. Un damier d'embeddings peut réunir plusieurs morceaux
disjoints. La partition conserve les pixels, sans garantir que chaque groupe
corresponde à un objet physique. Les réductions d'embeddings et de déplacement
se font enfloat64 ; embeddings bornés à[-1,1], déplacement représentable enfloat32.

Le descripteur contient49floats : centre et dimensions de la boîte(4), aire
relative(1), déplacement prédit relatif à l'écran(2), embeddingmoyen(32),
contrôlabilité et solidité moyennes(2), histogrammeRGB en huit octants(8).
Le déplacement porte sur l'intervalle prédit entre les deux observations,
**pas sur une seconde ni une frame native** tant que la collecte n'a pas fixé
son intervalle. Un histogramme est une lecture des pixels, pas un décodeur
de palette propre à un jeu.

L'identifiant retourné est un numéro dans l'image courante. Aucun lien temporel,
catégorie, compétenceME ou apprentissage de segmentation n'est établi ici.
Les tests synthétiques vérifient les dimensions, les coordonnées, les splits,
la conservation des petits objets, les limites et l'absence d'alias mémoire.
Le suivi, les objectifsC/G/I et leur validation demeurent à implémenter.
