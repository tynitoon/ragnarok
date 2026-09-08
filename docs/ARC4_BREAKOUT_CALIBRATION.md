# Calibration brute Breakout — développement

Protocole et premier candidat fixés au commit `95bcaab`, avant les 4 096 frames
de développement17111. Six épisodes à construction matérielle fixe, 4 888 frames
facturées intégralement. La séquence réservée17112 n'est pas utilisée pour les
corrections. Objets : raquette, balle et segments horizontaux de briques adjacentes,
par rangée de couleur ; murs permanents et HUD exclus. Les segments ne sont pas
108 sprites visuellement séparés. Les données restent côté évaluateur.

La baseline conservée dans `85dd5d8` échoue sur la balle : présence98,4375 % et
géométrie98,9113 %, contre99 % requis dans les deux cas. Raquette et six rangées
passent. Le décodeur maintenait la boîte entière de la balle jusqu'à des positions
où le rendu avait déjà disparu ou était tronqué au bas de l'écran.

La correction v1 est une approximation empirique de ce clipping, développée
uniquement sur17111. **Ce n'est pas une frontière physique universelle :** la ligne
y195 demeure visible dans11 des28 passages analysés, et disparaît dans17.
Ces11 faux négatifs restent dans les31 désaccords de présence de la balle ; aucune
règle RAM additionnelle n'est ajustée à ces cas. Les contacts de la balle et de
la raquette de même couleur restent aussi des limites du checker pixel inchangé.

Résultat corrigé : présence balle4065/4096 =99,2432 % ; boîtes à un pixel près
3018/3020 =99,9338 %. Raquette99,6094 % de boîtes valides, présence100 %. Les
six rangées passent (cinq exactes à100 %, dernière≥99,90 % sur les deux mesures).
Les seuils et tous les dénominateurs du protocole initial sont conservés.

La revue indépendante confirme le matching par cardinalité avant distance,
l'indépendance du checker pixel et la portée limitée de l'approximation. Dix
tests de géométrie/assignation passent, ainsi que trois tests du lecteur : anciens
coûts incomplets, transitions d'épisode absentes et corruption de shards rejetés.
La version revue doit être scellée avant la validation17112. Aucun gate, score,
événement, perception max-poolée ou apprentissage Breakout n'est validé ici.
Les quatre premières rangées sont restées intactes dans le développement :
leur comportement après destruction n'est pas couvert par ces trajectoires.
Le validateur impose aussi seed, mode, sticky, répétition, longueur et convention
déclarés avant de mesurer la séquence réservée.

## Validation réservée17112

Candidat scellé dans `6e48ef4`, puis capture de4 096frames,6épisodes et4 888frames
facturées. Les neuf sources scellées sont inchangées, contrôle imposé par le
calibrateur. **PASS pour tous les types**, avec les mêmes critères : balle
4064/4096 =99,2188 % de présence et3139/3141 =99,9363 % de boîtes à un pixel ;
raquette100 % de présence et4078/4096 =99,5605 % de boîtes valides.
Les six rangées passent également. Résultat conservé dans
`craft_v6_out/v70_breakout_calibration/validation_result.json`.
Cette validation conserve la portée et les limites ci-dessus ; aucune correction
du candidat n'est effectuée après ouverture de17112.

## Référence brute, protocole déclaré avant ses sorties

`reference_validation.json`, règle de mesure fixée dans `e83d862` : rappel des
centres mobiles5457/5501 =99,2001 %, balle3114/3139 =99,2036 %. La précision
est7060/9753 =72,3880 %, soit2693 prédictions sans correspondance. Tous ces
comptes sont conservés. Les segments de briques participent aux correspondances
visibles ; leur destruction n'est pas comptée comme un déplacement.

La référence ne reçoit que les RGB ordinaires et garde son historique entre
épisodes. La revue indépendante confirme les dénominateurs, l'appariement global
et l'absence de reset oracle. Quatre tests de scènes à plusieurs instances passent.
Le débit du diagnostic brut est≈577frames/s, hors émulation, échanges et branches.
Ce rappel élevé selon le juge calibré ne valide ni la qualité des boîtes prédites,
ni toutes les briques, ni un gate ou une perception apprise.
