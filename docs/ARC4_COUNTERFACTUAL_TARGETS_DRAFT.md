# ARC 4 — cible de contrôlabilité issue des propres expériences

Le prototype compare les images réelles de deux branches d'une même expérience.
Le choix brut/poolé est explicite. Il prend le déplacement du parser pour
l'horizon déclaré et projette les pixels des propres instances à cette position
future. C'est là qu'il échantillonne la différence des deux branches, avec
interpolation bilinéaire ; il n'utilise pas seulement l'emplacement initial.

La cible souple est la moyenne de cette évidence sur les pixels source dont la
destination reste visible. Elle supervise la têteC à ces pixels source. Une
instance est admissible si sa norme moyenne de déplacement prédit dépasse
0,25pixel. Les pixels à destination invisible, les instances sans mouvement
prédit et les branches trop courtes pour atteindre l'horizon sont inconnus :
leur poids est nul, pas une étiquette de non-contrôlabilité. La perteBCE sans
aucune cible admissible donne un gradient nul.

Cette cible peut être fausse : un autre objet peut changer à la position prédite,
un mouvement prédit peut être mauvais, le pooling conserve des pixels et les
actionssticky peuvent retarder un effet. Les conséquences indirectes peuvent
aussi diverger. Le prototype ne décide donc niME ni le délai du contrôle.
La procédure n'est ni un décodeurRAM ni un enseignant classique.

Le collecteur doit garantir que les deux traces viennent du même appel de
branche, que le déplacement prédit correspond à cet effet et à ce même horizon,
et que ces images appartiennent aux données autorisées pour l'agent. Des tableaux
de pixels ne prouvent pas cette provenance. Ce contrat n'est pas encore intégré
au collecteur ; aucune capture d'instrumentation n'a été utilisée ici.

Les tests portent sur des données synthétiques : déplacement rapide, demi-pixel,
support hors cadre, branches0/1/3frames, gradient d'une cible positive et gradient
nul d'une cible inconnue. Aucun résultat de perception ou de jeu n'est établi.
