# Candidate de perte P à plusieurs échelles spatiales

Prototype séparé préparé après les contre-exemples analytiques et avant le
résultat du diagnostic d'ajustement synthétique à deux scènes. Aucun ancien
parser, objectif, protocole ou résultat n'est modifié. Cette fonction n'est
branchée dans aucun entraînement.

La candidate compare la reconstruction du modèle à l'image future observée,
avec trois termes L1 de poids égaux : images originales, puis moyennes locales
de largeur3 et7. Les deux images reçoivent exactement le même filtre, de pas1,
avec bordure nulle incluse dans le dénominateur. Le terme original reste présent.
Le filtrage porte sur la perte, pas sur les observations du parser ou sur un
détecteur d'objet. Il ne fournit ni déplacement calculé, ni masque, ni identité,
ni arrière-plan classique. Ce choix reste une hypothèse d'optimisation.

Les tests synthétiques examinent la famille des translations du carré déjà
décrite : ils vérifient si un petit déplacement reçoit un gradient dans le sens
du futur lorsque le seul terme natif reste plat. Un tel gradient est une
propriété locale de cette fonction ; il ne démontre pas que le réseau apprendra
les bons déplacements, évitera toutes solutions dégénérées ou suivra un objet
réel. Le filtrage peut aussi gommer des détails ou introduire des effets de bord.

Une éventuelle comparaison d'ajustement devra annoncer son budget, ses termes,
ses graines et son critère avant la première mise à jour. Le diagnostic original
P natif est conservé avec son budget et ses résultats, qu'ils soient positifs ou
négatifs. Aucun critère de gate Atari n'est changé par ce prototype.
