# Asteroids — calibration externe, développement initial

Protocole `ARC4_ASTEROIDS_RAW_PROTOCOL.json` : 8 192 images ordinaires,
développement17511, réservé17512 fermé. Trois catégories séparées : vaisseau,
astéroïdes et tirs du joueur. Les critères restent 99 % pour les comptages
globaux et positifs et les boîtes à un pixel près, avec au moins50appariements
par catégorie. Aucun dénominateur vide ne vaut réussite.

La baseline fonctionnelle reprend des hypothèses des sources MIT OCAtari
épinglées au commit467accd5f3da379d83705b008094ffb787eec8a1 :
[RAM](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/ram/asteroids.py)
et [vision](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/vision/asteroids.py).
Ce décodeur n'est pas présumé juste. Sa condition de largeur5 impossible est
signalée ; la baseline conserve sa largeur6 effective. L'interprétation des
fragments et des couleurs partagées entre tirs et acteurs est une hypothèse
à confronter aux images, pas une exclusion des situations difficiles.

Revue indépendante préalable close : trois tests, les256encodages horizontaux,
les256orientations et120cas d'allocation/taille/visibilité vérifiés. Le reçu
`precollection_review.json` lie les sources et le protocole ; il permet seulement
la collecte de développement après commit. Aucun résultat de jeu ni de transfert
n'est établi. RAM, palettes et labels restent réservés aux instruments externes.
