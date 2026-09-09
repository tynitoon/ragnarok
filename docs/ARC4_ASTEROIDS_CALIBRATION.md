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

Les diagnostics distincts précisent le rendu : RAM57bit0 alterne les images des
rochers et celles du vaisseau/tirs. Pour les rochers,224 est une sentinelle ;
les autres valeurs hautes de y peuvent représenter une position négative. Les
deux groupes de registres ont un décalage vertical et les trois tailles ont des
enveloppes différentes de V0. Les glyphes complets ont été mesurés sur les seules
composantes de développement non coupées, puis la candidate est scorée sur
toutes les images, y compris les sorties, collisions et éclatements.

`rock_raster_development_v1.json` reste **négatif** :3838/4096comptages positifs
égaux,25966/26368boîtes à un pixel près. Les pièces de bord sont regroupées par
connectivité à travers x159↔x0 ; la représentation reste une boîte par morceau
visible, sans prétention d'identité physique persistante. Une nouvelle analyse
des erreurs distingue désaccord de pixels et désaccord de regroupement.

Le premier diagnostic indépendant vaisseau/tirs annonçait un passage, mais la
revue parent a détecté un défaut : RAM et RGB conservaient des pixels ày194 et
au-delà, alors que la borne gelée194 est exclusive. Ce résultat et son candidat
sont conservés comme **non conformes au périmètre**, avec correction séparée
et nouveau calcul requis. Il ne constitue pas encore un PASS conforme du jeu.
Réservé17512 toujours fermé.


La correction du ROI a été recalculée dans des fichiers distincts :
`asteroids_player_roi_v70.py` et `player_render_roi_development_diagnostic_v2.json`.
Sur toutes les8192images conservées, les comptages des deux catégories sont
exacts. Vaisseau :2309/2309boîtes exactes ; tirs :966/966à un pixel près,
964exactes. Quatre pièces visibles uniquement ày194 sont désormais hors ROI,
mais leurs images ne sont pas exclues. Onze tests passent ; le reçu
`player_missile_roi_correction_review.json` relie explicitement le résultat
initial non conforme et sa correction. Le jeu reste incomplet : les rochers
n'ont pas encore passé les critères.
L'inventaire complet `rock_raster_error_inventory_v1.json` conserve258images
avec désaccord de comptage et335avec au moins une erreur géométrique.
Sur279images en erreur, l'union des pixels prédits est pourtant exacte :
le diagnostic doit distinguer formes, occultations et regroupement des objets.
