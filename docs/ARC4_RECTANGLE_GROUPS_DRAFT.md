# Groupes du juge calculés depuis les rectangles

Candidate CPU séparée du 10 septembre 2026. Aucun moteur natif, décodeur,
référence ou score gelé ne change. Cette fonction n'est pas raccordée au
collecteur et ne fournit aucune donnée à l'agent. Aucun débit n'est encore mesuré.

Le groupement original remplit un masque pour chaque catégorie, puis étiquette
ses composantes connexes à quatre voisins. Les éléments de ce masque sont des
rectangles entiers déjà fournis par le juge. La candidate construit directement
leur graphe de connexion, puis prend les enveloppes de ses composantes.

Deux rectangles ont des pixels communs ou voisins si leurs intervalles se
recouvrent strictement sur un axe et se recouvrent ou se touchent sur l'autre.
Un seul contact par le coin ne les connecte pas. Chaque rectangle est lui-même
connexe ; les composantes du graphe sont donc exactement celles de leur union
pixellisée. Un balayage par ordonnée ne conserve que les rectangles antérieurs
dont le bord inférieur atteint encore la ligne courante. Une union transitive
construit les composantes. Le pire cas reste quadratique en nombre de rectangles.

L'ordre des catégories, les doublons, les ensembles d'identifiants et les boîtes
doivent être exactement ceux de l'original. Pour reproduire l'ordre de balayage
du raster, chaque composante est classée par son premier pixel occupé `(y,x)`.
Le coin supérieur gauche de sa boîte englobante ne suffit pas : le rectangle
le plus à gauche peut commencer plus bas que le premier pixel de la composante.
Un test distinct protège ce cas. Les identifiants de rendu restent des données
externes du juge ; leur conservation ne garantit pas une identité physique.

Les tests comparent les sorties complètes à la définition raster gelée :
contacts par arête et coin, recouvrements, chaînes, ordre, doublons, répétition
d'identifiants et scènes aléatoires déclarées dans les deux jeux du registre.
Ils n'utilisent ni capture native ni temps d'exécution comme critère.

Avant adoption, il faudra une revue indépendante, une vérification hors ligne
sur des entrées natives déjà conservées, puis un protocole distinct pour la
fidélité et le débit de la chaîne. Les résultats des anciennes variantes cache
et gather restent conservés. Une baisse du coût de ce seul composant ne
démontrera pas un gain de chaîne et ne changera aucun seuil du gate proposé.
