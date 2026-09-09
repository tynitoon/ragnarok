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


V2/V3 précisent l'unité mesurée : des pièces visibles par allocation, pas des
astéroïdes physiques ni leurs identités persistantes. La RAM garde les allocations
séparées et prédit les supports du premier groupe devant ceux du second. Les
fragments isolés restent des objets évalués. V2 atteint3905/4096comptages positifs
égaux et26215/26460boîtes within1 ; géométrie PASS, comptage FAIL.

V3 `asteroids_deblend_v70.py` traite les contacts de même couleur uniquement
lorsque deux glyphes complets expliquent exactement le masque et que toutes
les solutions donnent les mêmes supports visibles. L'origine du glyphe complet,
paire ou impaire, donne la priorité ; le bord visible ne suffit pas.97cas sont
démêlés, une ambiguïté reste inchangée. Résultat :4001/4096comptages positifs,
26436/26600boîtes within1, donc encore FAIL. Quatre tests passent. Le reçu
`rock_visible_piece_scope_review.json` confirme le périmètre et conserve trois
pixels résiduels de l'hypothèse V2 en image531.

V4 `asteroids_history_v70.py` utilise seulement la RAM courante et celle du
pas précédent dans le même épisode. Le premier P0 peut être placé avant sa
mise à jour horizontale ; les autres positions restent courantes. Un flag de
destruction nouvellement inscrit peut suivre le chargement du glyphe solide.
La correction explique exactement les témoins517,531,727,1851,2061,2131 et213,
y compris les trois pixels de531. Résultat complet V4 :4011/4096comptages
positifs,26497/26607within1 dont26478exacts. Comptage encore FAIL, sans exclusion.
V5 ajoute séparément les glyphes coupés aux frontières physiques et au ROI,
avec accord exact et unique des supports. Onze tests chronologie/bords passent ;
le résultat complet V5 est en cours.17512 reste fermé.


V5 est complète :8161/8192comptages globaux et4065/4096comptages positifs égaux,
26628/26673boîtes within1 dont26610exactes. Les31images en désaccord restent
incluses ; les débris d'explosion et certains contacts ambigus demeurent des
limites du modèle. La composition `development_visible_v1.json` reproduit ces
compteurs, ainsi que2309boîtes joueur exactes et966tirs within1 (964exacts).
Les trois catégories passent sur le développement. Revue indépendante close,
26tests de modèles et14tests de validation passent. Le protocole distinct
`ARC4_ASTEROIDS_VISIBLE_VALIDATION_PROTOCOL.json` définit le test17512 unique.
Le lanceur/scorer a été revu ; les commits des sources puis du scellé et le vrai
préflight sont encore requis avant construction. Aucun apprentissage ni résultat
réservé n'est déduit de ce passage en développement.


Le réservé17512 est complet et revu : source73fdf4c puis scellé158a976 committés
avant la première construction à11:58:47Z. Préflight réel conforme,31sources,
8métadonnées et runtime figés.8192images,8852frames facturées,5loads.
Joueur2051/2051boîtes exactes et tirs918/918exactes, comptages tous égaux.
Rochers :4073/4096comptages positifs,27541/27577within1 (27524exacts).
Les23images de comptage en désaccord et36paires au-delà1pixel restent dans
le résultat `reserved_visible_v1.json`. Le reçu `visible_reserved_review.json`,
SHAd019a899…, vérifie32archives, comptabilité, boutons fixés, chronologie des
commits, fractions et7témoins, dont4écarts reproduits. Les trois catégories
passent le protocole réservé sans nouveau réglage. Calibration brute close
pour les pièces visibles définies ; suite temporelle/référence distincte.

Le protocole temporel17521/17522 est préparé :8192décisions repeat4, boutons
uniformes tenus4décisions, budget complet et trois RAM adjacentes par fenêtre.
La définition utilise les rectangles sources remplis et la connexité4, avec
le ROI[16,194) inchangé. Les deux images sont décodées indépendamment avant
sélection des pixels contributeurs, sans RAM dans le juge RGB.
17tests passent, revue indépendante close sur41sources : empreinte
973aafd44d8f76cf10f835eb84fd157da0bf4718f35c5ad8ea641a3b2b3c2776.
Des témoins alternent correctement rochers et vaisseau/tirs ; le témoin négatif
brut53–54 est conservé (5RAM/18RGBrochers). L'instrument ne prétend donc pas
avoir déjà passé la nouvelle calibration. Commit et préflight réel avant17521.
Sortie de score prévue `pooled_geometry_development_v0.json`, pour distinguer
son marqueur d'échec de celui de la capture `pooled_development_v0_attempt.json`.

Collecte17521 complète après commit313b8d3 et vrai préflight conforme :8192fenêtres,
34476frames facturées,13loads,27,06s. Scoring V0 **incomplet** : garde
« Unknown player graphics pointer/direction » à la décision5320 ; le marqueur
failed est conservé, aucun résultat chiffré complet ne doit en être déduit.
L'inventaire de toutes les16384images sources trouve trois exceptions, current
5320/5321/5322 : explosion ày=-2, pointeurs172/178/184 au lieu de175/181/187.
Le décalage3 correspond à5+y, déjà prévu dans le modèle du vaisseau normal qui
traverse le bord. Sur une copie RAM, son application à l'explosion reproduit
exactement les deux boîtes visibles de chacune des trois images, ày18et192.
Adaptateur temporel V1 distinct `pooled_asteroids_wrap_v70.py`, cinq tests PASS ;
autres gardes et juge RGB conservés. Recalcul complet requis ;17522reste fermé.

V1 complète sur17521 : les trois catégories passent. Comptages positifs
joueur4357/4388, rochers8165/8192, tirs1876/1882 ; géométrie within1 respectivement
4415/4419,53018/53075,1900/1900. Les31/27/6 fenêtres de comptage erroné restent
incluses. Résultat `pooled_geometry_development_v1.json`, SHA768e4dcf… ; revue
indépendante `pooled_development_result_review_v1.json`, SHAedfdff3b… :64archives,
8192fenêtres, ledger et44sources vérifiés, dix témoins reproduits. Sources V1
committéesa2fccaf avant calcul. Le reçu commun de revue des corrections V1 a été
créé pendant le calcul : ce décalage est consigné, aucune revue avant calcul
n'est revendiquée. Le protocole de développement exigeait le gel avant réservé.

Validation17522 préparée séparément :49sources et9métadonnées doivent être
committées, puis le scellé distinct doit l'être avant construction. Revue du
préflight SHA5bd30d86… close ;30tests pertinents passent. Elle a fait renforcer
les bornes et identités des dénominateurs et la présence des métadonnées au
commit annoncé dans le scellé. Tentatives de capture/scoring distinctes et
durables ; aucun second essai. Aucun accès17522 à cette étape.

Réserve 17522 achevée après le commit du scellé f9217e3. Les 8 192 décisions
contiennent 8 182 avances de quatre frames et dix de deux, soit 32 748 frames
ordinaires + 14 × 132 de démarrage = 34 596 facturées. Les trois catégories
passent : comptages positifs joueur 4199/4217, rochers 8168/8192, tirs 1922/1924 ;
géométrie within1 respectivement 4231/4232, 53195/53249 et 1931/1931.
Les 18/24/2 fenêtres de comptage erroné et 1/54/0 erreurs géométriques restent
dans `pooled_reserved_geometry_v1.json`, sans nouveau réglage.

La revue `pooled_reserved_result_review_v1.json`, SHA15b8931c…,
confirme les 64 archives, le RNG et les actions, les 49 sources et la chronologie.
Ses 33 fenêtres témoins reproduisent 25 désaccords stockés et sept explosions
aux positions y=-1 et y=-4, absentes du développement : toutes leurs boîtes
joueur sont exactes. Aucun y=-2 dans la réserve ; l'audit local reste borné.
Calibration temporelle close. Les futurs diagnostics de référence utiliseront
le juge source corrigé sous une limitation explicite : ce résultat temporel
ne constitue pas une nouvelle validation brute indépendante.
