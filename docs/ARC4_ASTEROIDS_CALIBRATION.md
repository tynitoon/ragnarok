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

Collecte17511 exécutée après commit `8cbc41d` et vérification des13sources :
8 192images, cinq constructions froides,8 852frames facturées. Baseline négative
pour les trois catégories ; réservé17512 toujours fermé.

| Catégorie | Comptage positif égal | Boîtes à un pixel près | Statut |
|---|---:|---:|---|
| Vaisseau | 2183/4650 | 1790/2253 | FAIL |
| Astéroïdes | 1265/8192 | 9121/20974 | FAIL |
| Tirs du joueur | 926/2004 | 949/952 | FAIL |

Le premier export JSON a échoué après le calcul complet : des coordonnées
NumPyint64 dans les exemples n'étaient pas sérialisables. Le fichier partiel
`development_nominal_v0.json` et le reçu de cet échec sont conservés. Une
conversion vers les entiers Python, sans changement de géométrie, permet
l'export distinct `development_nominal_v0_serialized.json`. Les statistiques
sont exactement celles du préfixe complet du premier export. Le défaut de
sérialisation avait échappé à la revue préalable.
