# Disponibilité du signal : sonder les commandes avant de choisir un apprenant

Proposition du 10 septembre 2026, après l'inventaire `f04d837`. Ce document
prépare une expérience distincte ; aucun moteur, collecte ou ajustement n'est
lancé par cette note. Un protocole exécutable et sa revue restent à établir.

## Question et résultat qui motive la suite

Les six anciennes paires avaient la même commande 5 contre NOOP 0, à une seule
ancre par jeu. Cinq sont exactement identiques sur les huit futurs, bruts et
poolés. Asteroids fournit une union de 20 pixels différents à partir du quatrième
futur. Les trois autres cas ne contenaient aucune paire. Il serait donc trompeur
de traiter ces archives comme une supervision positive diversifiée. Cela ne
mesure ni la fréquence du signal dans les jeux, ni son origine physique, ni la
capacité d'un réseau à l'apprendre. La revue indépendante du résultat est en cours.

La question suivante est descriptive : avec toutes les commandes de l'interface,
à deux moments déclarés de jeu ordinaire, quelles conséquences RGB sont
observables à court terme ? Les commandes et moments doivent être fixés avant
de voir les nouvelles images. Aucun oracle, recherche de scène favorable ou
condition fondée sur le score, les vies, la RAM ou les objets du juge.

## Population proposée et coût explicite

- Les mêmes six jeux de développement ; aucun des quatre jeux réservés du gate.
- Deux ancres par jeu, après 10 000 puis 20 000 **images ordinaires effectivement
  livrées**. Ce compteur propre additionne les longueurs des traces reçues,
  sans convertir les resets froids ni les restaurations en images vues.
- À chaque ancre, commandes 0 à 17 contre NOOP 0, huit images demandées par
  branche. L'ordre de commandes est déclaré à l'avance. La paire 0 contre 0 est
  un témoin payé de répétabilité, pas une deuxième observation gratuite.
- Chaque paire restaure immédiatement la même ancre et conserve les deux
  préfixes réels. Aucun chargement additionnel d'une ancienne ancre éloignée.
- Une paire coûte au plus `8 + 8 + 2 × 60 = 136` frames expérimentales facturées.
  Dix-huit paires coûtent 2 448 ; au quota de 20 %, il faut au moins 9 792 frames
  ordinaires pour cette allocation. Les 10 000 images livrées entre les ancres
  la couvrent sans compter sur le crédit des initialisations.
- Deux séries coûtent au plus 4 896 frames expérimentales, pour au moins 20 000
  images ordinaires. Un plafond global par moteur doit inclure les constructions
  froides ; **30 000 frames facturées par jeu** est la proposition à vérifier,
  pas une permission de poursuivre après épuisement. Total maximal proposé
  180 000, dont au plus 29 376 frames expérimentales sur les six jeux.

Le quota reste vérifié par le moteur avant chaque paire. Les actions ordinaires
doivent suivre un calendrier commun aux jeux, avec sa graine et ses durées
fixées ; leurs choix ne dépendent pas de la disponibilité observée du signal.
Le franchissement d'une ancre peut dépasser le compteur cible de trois images
au maximum à cause du repeat 4 ; ce décalage est enregistré. Une frontière
visuelle ou un terminal ne déplace pas l'ancre pour en trouver une meilleure.

## Sorties et limites à geler

Conserver les 36 essais prévus par jeu, y compris les paires identiques, préfixes
courts et absences après arrêt. Pour chaque brut et poolé : nombre de pixels
différents à chaque horizon, union, premier changement, et longueurs appariées
réelles. Aucun seuil de signal positif, score global d'apprentissage ou cible
d'objet n'est sélectionné depuis ces mesures. Zéro au témoin 0/0 vérifie la
répétabilité observée ; un écart bloque l'interprétation des autres différences
et déclenche une investigation de l'instrument, sans effacer le résultat.

Toutes les commandes sont déclarées et tentées dans l'ordre prévu tant que
l'instrument et son budget le permettent. Une absence de signal n'autorise ni
prolongation, nouvelle graine, troisième ancre ni changement de boutons. Une
erreur après paiement conserve les comptes disponibles et bloque les actions
suivantes. Les tentatives, archives et résultats sont exclusifs et persistés.
Le plafond de temps CPU et de stockage sera fixé avant la collecte ; aucun
débit comparatif ne sera déduit de cette collecte. Pas de modèle ni de GPU.

Cette population serait un diagnostic de développement déjà observé. Elle ne
deviendrait pas après coup un jeu de validation indépendant pour une architecture
choisie depuis ses résultats. Une expérience ultérieure de prédiction devra
déclarer ses propres séparations, contrôles à commande constante, mesures sur
les pixels positifs et règles d'arrêt avant l'ajustement. Elle ne pourra pas
revendiquer le gate ou contourner ses plafonds en présentant ce diagnostic
comme de l'expérience gratuite. Aucun entraînement ultérieur n'est gelé ici.

Une réponse à une commande peut être indirecte ou se produire sur un fond
dévoilé. Même avec de nombreuses paires, les masques restent aux coordonnées
futures, sans correspondance physique avec une instance au départ. Les balles
non affectées, les occultations, m/C, ME et l'alignement des effets entre jeux
restent des problèmes distincts. Le test ne démontre pas le transfert.
