# Inventaire du signal RGB des branches déjà payées

Préparation prospective du 10 septembre 2026. Aucun nouvel entraînement, modèle,
émulateur ou accès aux tableaux réels n'est effectué pendant cette préparation.
Les anciens fits P restent négatifs et leurs sources et critères inchangés.

La revue du fit à priorités a établi une contraction des pixels blancs, puis
leur dilution par les contributions noires. Cela ne prouve pas un minimum local
mathématique, ni une impossibilité d'apprendre. Une reconstruction correcte ne
résoudrait pas non plus l'identité physique des pixels uniformes ou occultés.
L'autre constat conservé est celui du diagnostic CPU des objectifs : six paires
complètes et 150 instances propres, mais zéro pixel supervisé pour C. Ce résultat
ne distingue pas absence de contraste RGB et suppression par les gardes du parser.

## Population fixe et accès aux données

L'inventaire prend exactement les neuf cas du résultat natif owned : Pong,
Breakout, Freeway, Seaquest, Ms. Pac-Man, Asteroids, puis les trois cas Pong à
préfixe 1/2/3. Chacun des six premiers possède une paire de huit images contre
huit : commande 5 contre NOOP 0, ancre 576, époque visuelle 0, opération 145.
Les trois derniers ne possèdent aucune paire et restent explicitement présents.
Ils ne sont ni exclus, ni transformés en paires identiques à zéro changement.

Le résultat natif, sa revue, le diagnostic C et sa revue sont épinglés par SHA.
Leurs cartes `source_sha256` et `evidence_sha256` sont vérifiées séparément ;
un champ scalaire `result_sha256` ne doit pas être traité comme une carte.
Toutes les empreintes héritées restent vérifiées, sans importer leurs modules.
La fermeture exécutée ici est seulement le runner et NumPy ; les anciens modules
scientifiques dans les preuves sont des sources de provenance, jamais exécutées.

Le programme calcule les SHA des neuf fichiers NPZ et lit leur répertoire de
membres, sans décompresser les images ordinaires. Sur les six paires, il charge
exclusivement `branch_start_pair` et les six membres
`branch_[intervention/control]_[raw/pooled/endpoint]`. Les clés ordinaires
attendues sont contrôlées par leur nom seulement. Aucun membre RAM, objet, modèle,
récompense ou autre champ supplémentaire n'est accepté. Les trois archives sans
paire ne matérialisent aucun tableau. Total attendu : 42 tableaux de branches.

Les tableaux doivent être uint8, 210×160×3. Le max-pool de la première image
utilise le dernier écran brut de l'ancre et le premier brut de la branche.
Chaque max-pool suivant utilise les deux bruts réellement adjacents. L'endpoint
doit égaler le dernier max-pool ; dans le contrat synthétique de longueur zéro,
il égale le maximum des deux écrans de départ. Le runner réel exige les six
longueurs 8/8 gelées. Les fonctions pures couvrent aussi les préfixes pour
vérifier explicitement la sémantique de l'inconnu.

## Mesures descriptives, sans sélection

Pour chaque paire, chaque horizon de 1 à 8 et chaque représentation brute ou
poolée, un pixel est changé si `any(RGB_intervention != RGB_control)` est vrai.
On compte les pixels dont ce booléen est **True**, et non les canaux différents,
les intensités non nulles ou les valeurs uniques de l'image. Le dénominateur est
33600 pixels, HUD compris, sans ROI ou sélection d'objets.

Sont conservés : nombre et fraction changés, union cumulative et sa fraction,
histogramme du premier horizon changé, longueur commune et nombres de frames
non appariées. Le code 0 de l'histogramme signifie « jamais changé dans le préfixe
commun observé », pas « absence de contrôle », ni une prédiction au-delà de ce
préfixe. Un horizon manquant porte `observed=false` et des mesures/dénominateurs
observés `null`. Si aucune image n'est comparable, même le nombre « jamais changé »
et le statut de paire toute nulle restent `null`. Les paires présentes entièrement
nulles sont répertoriées séparément pour le brut et le poolé. Aucun seuil positif,
masque appris, correspondance, objet ou cible m/C n'est calculé.

## Barrières et arrêt

Sources et protocole doivent être committés, puis un préflight indépendant exact
doit être committé avant tout `np.load` réel. Le préflight vérifie le runtime CPU,
les sources, preuves, archives entières et blobs Git. `--check` ne charge aucun
tableau. Un marqueur exclusif est écrit, flushé et fsync avant l'ouverture de la
première archive. Une sortie ou tentative préexistante interdit tout retry.

Les empreintes d'archive sont contrôlées immédiatement avant et après leur
lecture ; l'intégralité du préflight est revérifiée après travail. Le résultat
est exclusif. Une erreur conserve la phase, le cas courant et les cas terminés,
puis bloque définitivement cette tentative. Le journal final est flushé/fsync ;
un arrêt dur peut laisser le marqueur `started` ou un journal partiel, jamais une
autorisation de reprise. Il n'y a pas de nouvelle dépense d'émulation à reconstituer.

Budget : CPU un thread, 300 secondes de travail, contrôlé avant/après chaque cas
et à la fin. Le délai est coopératif, sans interrompre une lecture NumPy en cours.
Les vérifications initiales et finales de preuves sont hors de cette durée.
Aucun Torch, ALE, GPU, optimiseur, inférence ou calcul de débit n'est nécessaire.
Un résultat `passed` certifie seulement l'exécution complète de cet inventaire.

## Proposition distincte de réponse visuelle future

Si cet inventaire établit un signal disponible, un prochain protocole pourrait
tester sa prédiction directement depuis les écrans passés, l'horizon et une
commande locale connue. La cible serait la différence des deux futurs observés,
aux coordonnées **futures**, sans filtrage par déplacement, m, priorité ou instances.
Le futur ne serait jamais une entrée de la prédiction ; le code local ne serait
pas assimilé aux douze effets sémantiques. Ni objectif, architecture, budget,
pondération de classes ou nouvel essai ne sont choisis par cet inventaire.

Le principe de contingence par comparaison des conséquences d'actions apparaît
chez Bellemare, Veness et Bowling (AAAI 2012), §4 : les restaurations permettent
de construire des cibles de régions affectées. Leur §6.2 utilise des situations
issues de trajectoires humaines ; l'implémentation emploie une représentation
par blocs de pixels. Ces données et choix ne sont pas repris ici, et leurs
résultats ne prouvent ni convergence ni perception réutilisable chez nous.
Source primaire : https://webdocs.cs.ualberta.ca/~mbowling/papers/12aaai-roc.pdf

Une réponse RGB peut être indirecte, concerner un projectile ou un fond dévoilé.
Une égalité observée pour cette paire et cet horizon n'est pas une preuve
d'immobilité physique ou d'absence de contrôle. m, C par instance et ME restent
inconnus. Les six ancres avec une seule commande ne permettent pas d'établir
diversité d'effets, généralisation ou transfert. Ces archives d'ingénierie déjà
utilisées ne deviennent pas un nouveau jeu de validation. Les balles sans réponse
à l'intervention restent un problème que ce signal seul ne résout pas. Les
critères, plafonds et règles STOP d'ARC4_VISION §2.1/§5 restent inchangés.
