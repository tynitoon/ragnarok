# Comparaison prospective des groupes sur deux archives déjà observées

10 septembre 2026. Une tentative hors ligne, sans nouvel émulateur, données
réservées supplémentaires, apprentissage ou mesure du débit de la chaîne.
L'environnement prévu est `.venv-arc4`, vérifié : Python3.14.3, NumPy2.4.4,
SciPy1.18.1, ALE0.12.1. Le module ALE peut être importé ; aucun moteur n'est créé.

Les entrées fixées sont les deux collections brutes complètes de MsPacman17471
et Asteroids17571, chacune de8192 images réelles déjà conservées, sur cinq épisodes.
Leurs manifests et chaque shard sont liés par empreinte. Tous les états RAM
restent dans cette vérification externe du juge, jamais dans une entrée de l'agent.

Chaque image brute est décodée une fois avec le décodeur antérieur inchangé.
À partir de la deuxième image de chaque épisode, les deux dernières listes
d'objets sont concaténées et envoyées aux deux méthodes de groupement. On vérifie
donc8187 paires par jeu,16374 au total. La première image de chaque épisode ne
produit aucune paire : aucun état froid précédent n'est inventé. Ce calendrier
consomme toutes les paires adjacentes disponibles ; il ne simule pas une commande
répétée quatre frames et n'est pas une trajectoire native supplémentaire.

Le critère est l'égalité exacte de toutes les sorties, dans leur ordre :
catégories, boîtes et ensembles d'identifiants. Une empreinte accumule les
entrées ordonnées et leurs groupes pour chaque paire. Toute divergence, mutation
d'entrée, collection incomplète ou erreur ferme la tentative et conserve son
préfixe. Aucun retry, sous-ensemble choisi ou autre graine n'est permis.

Les appels sont chronométrés individuellement, avec alternance globale : raster
d'abord sur les indices pairs, rectangles d'abord sur les impairs, sans remise
à zéro entre les jeux. Les comptes de premier appel sont conservés. La lecture,
le décodage et le calcul d'empreinte sont hors de ces temps de composant ; ils
font partie du temps total de travail. L'alternance réduit un biais d'ordre sans
constituer une estimation statistique contrôlée du gain sous charge ARC3.
Une équivalence réussie peut tout à fait être plus lente ; aucun seuil de vitesse
ne transforme ce résultat en verdict de gate.

Borne coopérative de300s de travail, vérifiée avant chaque ligne puis à la fin ;
les vérifications de preuves avant et après sont hors de cette borne. Threads
numériques fixés à1. Les sources, la revue indépendante de la candidate, ce
protocole et son préflight indépendant doivent être committés avant la lecture
des tableaux. Les tests de protocole utilisent seulement des lignes synthétiques.

Les expériences cache et gather précédentes restent inchangées. Ce résultat
n'autorisera pas à modifier un moteur actif ; un raccord natif séparé et sa
vérification seraient nécessaires avant toute mesure de chaîne.
