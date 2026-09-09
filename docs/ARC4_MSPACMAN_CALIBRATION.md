# Ms. Pac-Man : développement brut

Périmètre initial fixé et revu avant collecte, commit `9714014` : joueur,
fantômes, fruit, grosses et petites pastilles. Cinq catégories séparées,
ROI [0,0,160,171), aucun HUD. La clé ALE est `ms_pacman`. Revue Meitner :
grilles et palettes conformes aux sources MIT épinglées, 504 cas de bits isolés
vérifiés. Trois tests CPU initiaux passent.

Capture 17411 : 8 192 images ordinaires, quatre constructions froides,
8 720 frames facturées. Les 528 frames de préparation sont facturées mais ne
sont pas archivées individuellement. Réservé 17412 toujours fermé.

La baseline nominale V0 échoue sur les cinq catégories, conservée sans changement :

| Catégorie | Présence positive | Boîtes à un pixel près | Conclusion |
|---|---:|---:|---|
| Joueur | 100 % | 36,7188 % | Géométrie nominale incorrecte |
| Fantômes | 11,9629 % | 78,2251 % | Visibilité, contacts et géométrie à vérifier |
| Fruit | null | null | Absent, non calibré |
| Grosses pastilles | 0 % | 49,0853 % | Décor pris pour des pastilles et visibilité à vérifier |
| Petites pastilles | 49,5972 % | 99,2449 % | Comptages différents malgré beaucoup de boîtes proches |

Les petites pastilles représentent plus d'un million d'appariements, mais leur
nombre ne dilue pas les erreurs des acteurs. Aucune correction ni exclusion de
catégorie n'est tirée du réservé. Les prochaines corrections utilisent seulement
ce développement ; une collecte supplémentaire pour les objets absents devra
être définie avant son ouverture. Aucune perception apprise ni compétence de jeu
n'est démontrée par ces mesures d'instruments.

Les diagnostics du 9 septembre identifient des phases de rendu distinctes.
Le candidat séparé `mspacman_render_v70.py` décrit le joueur avec la position
et l'animation de la RAM précédente du même épisode : corps 8×10 à
(RAM10−12,RAM16+1), rogné de 2−RAM55 du côté RAM56 (haut/droite/bas/gauche).
Les quatre premières images, sans RAM précédente archivée, utilisent explicitement
la RAM courante. Vérification sur toutes les images de développement : 8 192
comptages égaux et boîtes à un pixel près, dont 7 913 exactes. Trois tests ciblés
passent. C'est une correction de développement, pas une validation réservée.

Les grosses pastilles suivent les flags117 et la phase30&8 **courants** ; leur
enveloppe nominale ne corrige pas encore les occultations. Les fantômes ont une
liste de rendu distincte de leurs positions logiques ; sa géométrie et le checker
RGB sont en cours de diagnostic indépendant. Les petites pastilles nécessitent
des silhouettes d'acteurs pour prédire les occultations sans consulter les pixels
dans le juge RAM. Le fruit reste absent et non calibré.

Les explorations sont conservées dans les JSON `player_phase_*`,
`player_drawlist_*` et `player_opaque_support_*`. Deux pistes sont rejetées :
RAM34/40/46 est constamment nul (aucune entrée joueur à cet endroit), et un
« nœud rouge » ne figure pas dans les crops propres examinés : le joueur et son
nœud sont jaunes. Le diagnostic de masque jaune|rouge peut être contaminé par
les fantômes rouges ; il ne constitue pas une silhouette opaque validée. Une
boîte correcte ne suffit pas à certifier la forme pixel par pixel.

Le candidat composé V1 a maintenant parcouru les 8 192 images uniformes avec
un juge RAM autonome et un checker RGB causal, initialement vide. Le résultat
`development_visible_v1.json` conserve les fractions par catégorie :

| Catégorie | Présence positive | Boîtes à un pixel près | Statut de développement |
|---|---:|---:|---|
| Joueur | 8192/8192 | 8192/8192 | PASS |
| Fantômes | 7785/7785 | 19562/19591 | PASS, 29 erreurs géométriques gardées |
| Fruit | null | null | Absent, FAIL/non couvert |
| Grosses pastilles | 8192/8192 | 15359/15497 | PASS, marge étroite |
| Petites pastilles | 8183/8192 | 1057309/1057309 | PASS, 9 erreurs de présence gardées |

Le résultat global reste donc **FAIL : fruit non couvert**. Aucun réservé n'est
ouvert. Ce passage de quatre catégories ne constitue pas encore une calibration
complète du jeu ni une mesure d'apprentissage.

Les fantômes utilisent la liste de rendu de la RAM précédente, jusqu'à quatre
entrées (255 signifie aucune), six couleurs dont le gris214 absent de V0.
Les motifs normaux et effrayés partagent une phase miroir donnée par le bit16
de RAM30 **courante**. Le juge des pastilles compose ces motifs et les douze
silhouettes jaunes du joueur ; il garde les pixels de pastille non couverts,
y compris les fragments à travers les yeux et les pieds des fantômes. Les neuf
erreurs de présence correspondent aux résidus déjà diagnostiqués du joueur.
La première version du module de support joueur est conservée octet pour octet
dans `player_only_support_source.py`, conforme au SHA du diagnostic antérieur.

Le checker des pastilles apprend ses emplacements uniquement depuis des
rectangles roses complets déjà observés, puis rassemble leurs fragments visibles.
Une grosse pastille garde priorité sur les petites formes qu'une occultation
pourrait créer en son sein. Sa variante causale est identique au diagnostic
rétrospectif sur cette séquence ; l'échec de la variante naïve reste consigné.

Collecte supplémentaire **préparée, pas encore exécutée** : protocole
`ARC4_MSPACMAN_COVERAGE_PROTOCOL.json`, développement17431, réservé17432 fermé,
32 768 frames ordinaires fixes. Le contrôleur cherche des pastilles depuis le RGB
seul ; aucun label RAM, récompense ou snapshot ne dirige ses actions. La revue
a corrigé deux défauts avant collecte : sauts de coût dans Dijkstra et immobilité
masquée par l'animation de bouche. Aucun résultat favorable ne déclenche un arrêt
anticipé ; le fruit absent de17411 reste explicitement non couvert.

La revue indépendante préalable est close : trois tests ciblés et deux témoins
de routage passent. Le reçu `coverage_precollection_review.json` lie les dix
sources de collecte, les tests et les résultats uniformes par SHA256. Il autorise
la collecte de développement après commit ; il ne garantit pas l'apparition du
fruit et ne valide aucun jeu réservé.

Collecte17431 terminée après commit `de0f7d3` :32 768images ordinaires,
33 560frames facturées,6constructions froides,182,42secondes. Le fruit est
maintenant observé. La calibration V1 s'interrompt à l'image1315 sur un état
d'animation joueur inconnu ; le diagnostic complet compte5787images concernées.
Le reçu `coverage_visible_v1_failure.json` conserve cet échec sans score partiel.

La RAM55/56 est un espace temporaire partagé avec le fruit. La liste spéciale
de rendu, distincte de la liste des fantômes, utilise RAM34 pour sa longueur,
RAM45/46 pour ses codes et RAM51/52+53/54 pour ses pointeurs. Elle explique
les images où le joueur ou le fruit clignote, malgré une position logique active.
Le nouveau module joueur séparé utilise son pointeur compensé par y pour choisir
la silhouette ; son diagnostic indépendant conserve les pixels résiduels.

Les diagnostics `fruit_list_development_diagnostic.json` et
`fruit_support_development_diagnostic.json` gardent la première hypothèse8×10,
qui échoue géométriquement pour la seconde forme de fruit. Les deux formes
observées sont distinctes : cerises8×10 à(x−12,y+1), fraise6×8 à(x−11,y+3).
La liste prédit exactement les10 677images avec fruit ;475silhouettes perdent
leur dernière ligne visible, résidu conservé. Trois tests du module RAM fruit
passent, dont les images où la liste affiche seulement le fruit ou le joueur.
La composition V2 sera scorée séparément sur17411 et17431 ; aucun réservé ouvert.

V2 uniforme17411 terminé : les neuf erreurs de comptage des petites pastilles
disparaissent,8192/8192comptages égaux et1 057 318boîtes toutes exactes. Joueur
8192/8192boîtes à un pixel près,8120exactes. Fantômes et grosses pastilles gardent
leurs fractions V1. Le fruit demeure absent/null/FAIL sur cette séquence.
La couverture17431 est encore en calcul. Revue de composition indépendante
`list_composition_preseal_review.json` close,25sources liées ; ce reçu ne tient
pas lieu de revue des résultats complets ni de validation réservée.
