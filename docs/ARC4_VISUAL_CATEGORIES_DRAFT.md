# ARC 4 — catégories visuelles locales

Prototype séparé, testé uniquement sur des descriptions propres synthétiques.
Le codebook reprend le voisin le plus proche de la vision ARC 4 : concaténation
de l'embedding moyen (32), des logarithmes des fractions de largeur et hauteur
(2) et de l'histogramme RGB propre (8), soit 42 nombres. Distance euclidienne
sans pondération implicite ; seuil proposé de 0,6, nouvelle catégorie au-delà.
Ces choix n'ont pas encore été gelés pour une expérience ni validés sur Atari.

Un objet observé met à jour la moyenne de sa catégorie avec un poids égal par
observation. Les absences maintenues par le tracker n'apportent aucune mise à
jour. Dans une image, les objets sont traités dans l'ordre de leur premier pixel,
indépendamment des numéros de pistes et de l'ordre de la liste. Les catégories
créées plus tôt dans cette image sont disponibles pour les objets suivants.
Les numéros sont locaux et ne portent aucune sémantique de jeu.

Le plafond est de 64 catégories. Une mise à jour qui dépasserait ce plafond est
entièrement refusée : ni fusion forcée, ni éviction, ni réécriture des moyennes
antérieures. Le futur protocole doit compter ces refus et fixer leur traitement.
Le nombre d'observations par image est aussi borné, à 512 par défaut. Aucune mesure
de coût ou de mémoire réelle n'est déduite de ces seules bornes.

Ce codebook doit être vide à chaque nouveau jeu, dans chaque bras. Son export
JSON décrit l'état local et reste exclu de l'héritage entre jeux. Ce n'est donc
pas encore une mémoire qui accélère l'apprentissage. La fusion selon les
comportements et la contrôlabilité, le registre d'interactions et les compétences
réutilisables restent à construire. Les animations, changements de palette,
bruits de segmentation et variations de taille peuvent créer trop de catégories ;
aucun nom d'objet ni vérité RAM ne vient corriger ce problème dans ce prototype.
