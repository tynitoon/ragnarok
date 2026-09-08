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
