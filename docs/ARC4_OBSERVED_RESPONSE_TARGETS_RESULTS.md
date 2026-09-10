# Réponse observée et attribution : interface vérifiée

10 septembre 2026. Les **27 fixtures passent**, en **3,3302088 s** d’exécution
totale, imports compris, sous les 60 s cumulées déclarées. Aucun réseau,
optimiseur, émulateur ou fichier de données expérimentales n’a été exécuté ou
ouvert. Le temps indiqué par pytest seul est de 2,56 s.

## Ce qui est maintenant séparé

`ragnarok/learning/observed_response_targets_v70.py` distingue l’entrée de
prédiction, limitée aux deux images passées et à l’hypothèse propre d’effet,
des cibles construites après observation d’un `BranchExperience`.

Chaque horizon conserve sa réponse RGB exacte lorsqu’elle est disponible dans
les deux branches. Une image manquante donne `None`, tandis qu’une comparaison
observée sans différence donne un masque booléen vide. Les traces brutes et
poolées restent deux choix explicites. La longueur commune, les queues non
appariées et l’origine enregistrée sont conservées.

La cible m couvre les pixels source même sans instance proposée. Chaque C reste
lié à un support d’instance propre. **Toutes les attributions m/C sont inconnues**,
avec valeur `None`, poids nul et motif explicite. Une réponse visible n’est
convertie ni en identité d’objet, ni en contrôle direct, ni en label négatif.

## Vérifications et correction issue de la revue

Le contrat 253a807 a été relu avant exécution. La revue de dd7abd1 a trouvé une
mutabilité résiduelle : des octets protégés ne suffisaient pas à empêcher la
modification de la forme ou du type d’un objet NumPy partagé entre horizons.
La correction c4b789f conserve uniquement des octets, une forme et un type
immuables, puis construit une vue indépendante à chaque accès. Le défaut a été
levé par une seconde lecture, sans calcul du réviseur.

Les fixtures couvrent une réponse non vide, un flash dont les comparaisons
brute et poolée diffèrent, les préfixes asymétriques ou absents, les 64 horizons,
les deux paires de mondes occultés de causalités opposées, les incohérences
d’ancre et d’effet, les traces invalides et les supports hors image ou partagés.
Elles vérifient aussi les mutations des entrées, des pixels et des métadonnées
de tableaux, ainsi que la stabilité des autres horizons. Les états physiques
des juges synthétiques ne sont jamais transmis à l’interface.

Il n’y a eu qu’une exécution des tests et aucun échec d’exécution. La correction
d’immutabilité a précédé cette exécution. Les sources du lanceur ont également
été committées avant leur utilisation ; reçus et sortie sont conservés en
2b7d08a dans `craft_v6_out/v70_observed_response_interface/` :
`check_v0_attempt.json`, `check_v0.json`, `check_v0_output.txt`.

## Ce que cette conformité ne démontre pas

Il s’agit d’un contrat de données, pas d’un résultat d’apprentissage. Le nombre
de cibles m/C admissibles reste nul. Les identifiants et égalités de tableaux
ne certifient ni paiement, ni replay natif, ni causalité physique ; le propriétaire
des opérations reste responsable de ces conditions et de la chronologie.

La dépendance entre reconnaissance de soi et découverte des effets demeure,
ainsi que l’incertitude de correspondance sous occultation. La suite doit
préciser quelle correspondance un mécanisme appris peut raisonnablement
estimer, avec quel domaine de validité, quelles erreurs et quelle couverture,
avant de raccorder un apprentissage. Aucun ancien objectif gelé, résultat,
critère de gate ou validation conditionnelle n’est modifié par ce jalon.
