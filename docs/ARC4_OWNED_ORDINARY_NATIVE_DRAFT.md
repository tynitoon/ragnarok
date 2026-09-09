# Raccord natif des images ordinaires, avant le modèle

9 septembre 2026. Préparation prospective seulement : aucun émulateur lancé.
Le script compare le collecteur possédant un worker natif au moteur froid
original, instrumenté par copie de ses lectures RGB déjà effectuées après
paiement. Le témoin ne fractionne pas les commandes et ne rejoue pas une image
gratuitement. Son observation de départ se fait après une éventuelle construction
froide. Aucun accès à la RAM, à une récompense, aux vies ou à une classe de sprite.

Six cas normaux couvrent les six jeux déjà employés en développement. Trois
cas Pong fixent à l'avance une limite d'épisode destinée à produire des préfixes
de une, deux et trois images. Tous les écrans de départ, images brutes, images
max-poolées et écrans finaux sont comparés, dans leur ordre, pour chaque commande.
Les deux chemins doivent avoir des comptabilités intégralement identiques.

Le témoin de chronologie calcule seulement les ruptures d'écran courant et les
longueurs effectivement livrées. Il n'utilise pas un événement natif comme
étiquette de mort. Une branche dans chaque cas normal doit conserver l'ancrage
et l'horloge ordinaires ; deux commandes après la branche vérifient la reprise.
La fidélité du détail RGB des branches dépend aussi du jalon natif précédent.

Huit positions ordinaires choisies avant exécution sont archivées par cas avec
leurs vrais tableaux RGB et métadonnées de commande propres. Ce sont des traces
d'ingénierie destinées à un futur essai de raccord sur CPU ; aucune entrée du
juge et aucun objectif supervisé externe ne sont ajoutés. Aucun entraînement,
checkpoint, gain d'apprentissage ou mesure de débit n'est produit ici.
Les six paires de branches déjà prévues et payées sont aussi archivées entièrement
avec leur départ : cela permettra de vérifier le raccord C sur CPU sans ajouter
une collecte. Cet ajout précède toute revue préalable ou exécution native ; il
ne modifie aucune action, trajectoire ou charge du protocole.

Une revue préalable exacte est obligatoire avant la première construction.
Toute tentative, dépense partielle et archive créée est conservée après échec.
Les rapports réussis sont mémorisés dès réception et ne sont jamais relus après
un échec d'un autre rapport. La perte d'un constructeur reste un coût inconnu
explicite. Les plafonds sont 2 000 charges par instance, 18 instances au total,
donc au plus 36 000 charges connues sur une exécution entièrement comptabilisée.
