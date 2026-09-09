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

V2 couverture17431 terminée : joueur et fruit passent les comptages et boîtes
à100 %. Le résultat global reste **FAIL** : fantômes31766/32117comptages positifs
égaux, grosses19260/20646, petites30678/32108. Les résultats uniformes et ceux
de couverture ne sont pas mélangés.

Les1386images du nouveau labyrinthe utilisent le bleu(101,111,228), encore
inconnu du checker rose. Les murs clignotent aussi en gris214 en fin de tableau
(ex4329), confondu avec des fantômes. Le passage horizontal des fantômes produit
une pièce visible à chaque bord (ex3798). Les grosses pastilles nécessitent
également l'occultation par les silhouettes d'acteurs.

Candidat V3 distinct : `mspacman_maze_visible_v70.py`. Registres RGB séparés par
palette ; murs appris depuis les grandes composantes déjà vues et retirés du
gris seulement pendant un flash couvrant95 % du masque appris. Les pixels gris
hors de ce masque restent analysés. Le juge RAM applique les supports bouclés
des fantômes et les occultations aux deux tailles de pastilles. Quatre témoins
ciblés passent. Les deux calculs complets V3 sont en cours ; aucune source de
ces calculs ne doit changer avant leur fin. Réservés17412/17432 fermés.


V3 complète sur les deux développements : uniforme17411 conserve le fruit
absent/null/FAIL et les quatre catégories observées passent. Couverture17431 :
les cinq catégories passent, sans mélanger les séquences. Les comptages sont
exacts sur toutes les images pour joueur, fantômes, fruits et grosses pastilles.
Les fantômes gardent704erreurs géométriques sur86 503appariements
(99,1862 % à un pixel près). Les petites pastilles gardent45désaccords de
comptage sur32 768images et21erreurs géométriques sur2 374 804appariements.
Les grosses pastilles ont27 954boîtes exactes. Ces résidus restent dans le score.
Reçus indépendants de composition et de résultats V3 clos. Le protocole distinct
`ARC4_MSPACMAN_RAW_VALIDATION_PROTOCOL.json` prépare les réservés17412/17432 :
scellé des sources et métadonnées avant lancement, deux séquences séparées,
fruit uniforme strictement absent toujours null/FAIL. Aucun réservé ouvert à
ce point documentaire.


La paire réservée a terminé sans changement de sources.31sources du commit
5a2ee619 et scellé committé1fddd2c avant toute construction ; préflight réel
PASS avant et après le calcul. Uniforme17412 :8192images,8852facturées,5loads ;
les quatre catégories observées passent, fruit toujours strictement absent
(null/FAIL). Couverture17432 :32768images,33824facturées,8loads ; les cinq
catégories passent séparément. Fantômes :31682/31914comptages positifs,
85316/85854within1. Petites :32464/32504comptages positifs,
2517375/2517439within1. Joueur, fruits et grosses ont des comptages exacts.
Le résultat `raw_reserved_pair_v3.json` satisfait l'acceptation prospective,
sans changer le FAIL individuel du fruit uniforme ni agréger ses dénominateurs.
Le reçu indépendant `reserved_pair_review_v3.json`, SHAa53c1655…, vérifie les
160archives,40960lignes comptables, sources, commits, dates et fractions ; le
scan RAM indépendant confirme10426fruits en couverture et0en uniforme.
Les erreurs restent consignées. Calibration brute close pour le périmètre
annoncé ; aucune exactitude universelle ni capacité d'apprentissage n'en découle.
La suite temporelle utilisera un adaptateur distinct à trois instantanés RAM,
car chaque image brute doit être décodée avec son prédécesseur.

Le protocole temporel distinct fixe17421développement et17422réservé,
8192décisions repeat4 avec le même contrôleur RGB de couverture. Les images
intermédiaires non archivées sont facturées ; plafond conservateur1114244frames.
Trois RAM adjacentes et leurs numéros sont archivés pour les deux RGB réels.
Le lecteur vérifie les troncations, recouvrements, démarrages froids et charges.
Chaque composante regroupe par connexité4 les rectangles sources remplis,
après reboîtage des pixels contributeurs côté RGB. Les cinq catégories gardent
les seuils séparés99% et50appariements minimum, sans exemption de fruit absent.
43tests passent. La revue a fait corriger le refus d'installation chaude et les
reçus d'échecs : budget partiel capturé lorsque disponible, coût constructeur
inconnu explicite sinon, tentative de scoring séparée. Ce dispositif reste
prospectif : sources et reçu doivent être committés avant toute construction17421.

Collecte17421 complète : source1dedc9f avant construction, préflight conforme,
8192fenêtres,32756frames ordinaires (8188avances4 et4troncations1),5loads,
33416facturées. Le score V0 conserve un **FAIL fantômes** :8080/8192comptages
globaux,7956/8068positifs ;27999/28065boîtes within1. Les quatre autres types
passent, avec3285fruits observés. Revue indépendante d4faf514… confirme les
preuves et ce résultat négatif. La première commande de score a été refusée
avant calcul car son nom de marqueur était déjà celui de la capture ; la sortie
distincte `pooled_geometry_development_v0.json` et son marqueur sont préservés.

Sur les témoins246/1250, une première ligne vide abaisse le bord RGB d'un pixel :
les rectangles RAM se touchent, les RGB restent disjoints. Aucun pixel n'est
supprimé par le max dans ces témoins. L'inventaire complet des sources17421
observe1557/1557premières lignes fantôme vides lorsque ymod12=1 ; les autres
écarts restent explicitement inventoriés. La candidate temporelle V1 distincte
`pooled_mspacman_rows_v70.py` retire cette seule ligne côté RAM ; RGB, catégories
non fantômes, connexité4 et seuils ne changent pas. Cinq tests de témoins passent.
Les anciennes sources brutes/temporelles restent intactes. Nouveau calcul
complet requis avant toute revue/gèle du réservé17422, toujours fermé.
