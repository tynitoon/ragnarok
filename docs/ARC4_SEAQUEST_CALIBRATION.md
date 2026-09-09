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
