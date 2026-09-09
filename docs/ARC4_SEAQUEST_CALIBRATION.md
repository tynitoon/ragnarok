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
