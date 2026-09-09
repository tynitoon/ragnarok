# Seaquest — préparation de la calibration brute

Le protocole `ARC4_SEAQUEST_CALIBRATION_PROTOCOL.json` fixe 8192 frames de
développement17311, puis validation17312 après revue et scellement. Six
catégories d'acteurs mobiles, toutes leurs instances : joueur, requins,
sous-marins ennemis (surface comprise), plongeurs, missiles ennemis et missile
joueur. Les jauges et les autres éléments HUD sont exclus de cette calibration.

Faits de coordonnées et palettes issus des modules MIT
[RAM Seaquest](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/ram/seaquest.py)
et [vision Seaquest](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/vision/seaquest.py)
au même commit épinglé. Le port RAM reste séparé du checker RGB par couleurs et
composantes. Les slots RAM sont consignés comme emplacements d'allocation ; ils
ne prouvent pas la continuité d'une identité après une réapparition.

La baseline peut échouer aux animations, contacts, fragments ou limites d'écran.
Les six catégories exigent chacune au moins50appariements et les seuils99%
déjà déclarés. Une catégorie absente ne sera pas créditée d'une calibration
parfaite : il faudra définir prospectivement une couverture supplémentaire,
sans ouvrir la validation pour régler le candidat. Aucun apprentissage ni gate.

## Baseline conservée

Protocole et code fixés `339704a` avant17311. 8192 frames, six constructions
froides, 8960 frames facturées. Baseline `31752a2` : aucun type ne passe.
Sous-marins et missiles ennemis n'apparaissent pas côté RAM et restent non
calibrés. Les petites bulles bleues sont faussement classées comme missiles par
le checker initial ; le détecteur primaire utilisait aussi largeur>2. Les
dimensions du joueur et du corps des plongeurs demandent une correction de
développement. Les bulles décoratives ne sont pas l'une des six catégories
d'acteurs préalablement déclarées. Elles ne doivent pas être fusionnées au corps
des plongeurs pour rendre artificiellement les boîtes plus hautes.

Validation17312 fermée. Préserver chaque résultat avant correction ; toute
nouvelle stratégie de couverture sera fixée avant de collecter ses données.

## Correction de développement v1, toujours négative

La bande y53:57 masque le joueur à la surface. Les fragments visibles doivent
être regroupés verticalement, sans agrandir les boîtes avec les pixels ajoutés
par cette opération. Le compteur de réapparition alterne aussi des frames
visibles et invisibles. Aux bords, les boîtes doivent suivre le support du
sprite : l'intersection de son rectangle nominal avec l'écran est insuffisante.
Les masques du joueur, des requins et des plongeurs ont été inférés sur 17311
uniquement ; ce sont des corrections empiriques du juge RAM externe.

Le protocole correctif V1 conserve les seuils et les résultats originaux.
`development_v1.json` échoue encore : boîtes joueur 98,349 % (baseline 29,031 %),
requins 97,303 % (84,730 %), plongeurs 99,744 % (60,152 %). La présence des
plongeurs reste à 93,274 % : les corps rognés au bord sont mal classés par le
checker RGB. Les contacts jaune sur jaune perturbent encore les missiles du
joueur. Aucun seuil n'est abaissé, aucune frame n'est exclue.

Revue indépendante Ohm : aucun drapeau de sous-marin ni missile ennemi sur les
8192 RAM, et compteurs de progression/sous-marin de surface toujours nuls. Le
filtre bleu largeur>2 retire 20502 petites composantes, mais en laisse 602 de
largeur 3 à 5 et hauteur 2 à 5. Celles-ci ne sont pas déclarées décoratives pour
faire disparaître les désaccords. La capture supplémentaire devra utiliser un
contrôleur pixels/boutons réservé à l'instrumentation, partir à froid et garder
toutes les frames, morts et coûts. Sa couverture sera évaluée séparément de
l'accord géométrique et ses résultats ne dilueront pas ceux de 17311.

## Couverture supplémentaire préparée avant 17331

`ARC4_SEAQUEST_COVERAGE_PROTOCOL.json` fixe 32768 frames et un contrôleur RGB
réservé à l'instrumentation. Il attend le remplissage visible de l'oxygène,
cherche les plongeurs, puis remonte après six prises ou lorsque l'air baisse.
Ces règles s'appuient sur le [manuel Activision](https://www.atariage.com/manual_html_page.php?SoftwareLabelID=424).
Il n'utilise ni RAM, ni récompense, ni accord du juge pour choisir les boutons.
Revue Ohm close après correction de l'attente initiale et ajout de l'identité
finale des coûts ; deux tests CPU sans nouvelle ROM passent.

17332 est déclaré réservé, mais son exécution reste entièrement désactivée
jusqu'à la revue et au sceau complet des sources de géométrie et de mesure.
Les 99 % de présence sur frames positives sont fixés pour la nouvelle collecte
avant son ouverture. Sous-marin de surface évalué séparément, pas de dilution
des erreurs anciennes, ni de confusion entre comptages d'objets par frame et
apparitions indépendantes. Les fixtures ne servent pas à entraîner l'agent,
la référence RGB ou à lui fournir des états avancés.

## Résultats de développement v2 et couverture 17331

Après le commit du protocole/captureur `9800cec`, 17331 a été collectée en entier :
32768 images, 14 chargements froids, 34560 frames facturées. Les sous-marins
immergés apparaissent sur 311 frames (989 instances-frame) et leurs missiles
sur 223 frames (546 instances-frame), tous dans un seul épisode. Aucun
sous-marin de surface : couverture globale toujours incomplète. Ne pas déduire
de ces comptages une calibration géométrique ou des apparitions indépendantes.

Sur l'ancien développement 17311, V2 passe pour joueur, requins et missiles du
joueur : respectivement 99,255 %, 100 % et 99,438 % de présence ; toutes les
boîtes appariées respectent la tolérance d'un pixel. Plongeurs et missiles
ennemis restent en échec et les deux catégories absentes restent non calibrées.
V2 regroupe les fragments de requins et sépare certains contacts jaune sur jaune
en ancrant le projectile à son extrémité exposée. Revue Ohm : 128 contacts
reconstruits, tous dans la palette visible, sans accès RAM dans ce chemin.

**Limite conservée** : au bord x8 ou x160, l'extrémité peut être elle-même coupée.
Un corps x10:26 et deux pixels exposés x8:10 peuvent correspondre aussi bien à
un missile nominal x2:10, visible sur deux pixels, qu'à x8:16, large de huit.
L'image est identique ; le checker choisit huit. Cette reconstruction n'est pas
une preuve de visibilité ou d'identité parfaite lors des occultations.

`coverage_geometry_v2.json` applique les critères supplémentaires prévus avant
17331. Joueur et requins passent ; sous-marins immergés échouent en géométrie
(61,881 %), plongeurs échouent en présence positive (78,064 %), missiles ennemis
en présence et géométrie. Missiles du joueur : toutes les boîtes appariées sont
exactes, mais présence positive94,292 %, donc échec. Surface vide, dénominateur
positif null, pas de succès fictif. Deux tests du scoreur vérifient qu'une longue
suite vide ne dilue pas les acteurs manqués et qu'une catégorie absente échoue.

La capture a conservé 52 morts ; le HUD lu dans les images concorde avec la RAM
sur toutes les32768frames et ne dépasse jamais deux plongeurs embarqués. Le
contrôleur poursuit les plongeurs horizontalement et tourne ainsi parfois le
dos aux requins. Une seconde tentative de couverture est préparée séparément,
avec maintien d'une position centrale et tir vers l'ennemi proche. Le premier
échec de couverture reste un échec ; aucun verdict ancien n'est réétiqueté.

La politique de couverture V2 et son protocole fixent17341/17342, à nouveau
32768frames. Revue Ohm : séparation/coût/seuils conformes, défaut d'orientation
au contact d'un missile corrigé en mesurant le corps hors de sa ligne. Trois
tests hors ROM passent, dont deux positions de contact qui doivent maintenir
FIRE sans avancer. La capture réservée reste désactivée. Chaque catégorie,
surface comprise, devra atteindre50appariements dans17341elle-même.

17341 collectée après `572151b` :32768images/32896frames facturées, un seul
épisode. Davantage de phases observées (RAM61 atteint2), mais RAM60 reste0 :
aucun sous-marin de surface, couverture toujours FAIL. À partir d'environ8000,
le contrôleur reste trop près de la surface : le centre de la partie visible
fait croire qu'il a atteint le niveau de recharge alors que son sommet est
encore y47. Le jeu exige le retour au sommet visible y46. Le contrôleur attend
donc sans finir la remontée. Ce défaut du contrôleur de collecte sera corrigé
dans une version distincte ; cette capture reste complète et conservée.
