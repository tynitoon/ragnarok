# Calibration affine : apprentissage de l’association sur le lot

10 septembre2026. Le fit affine est complet en46,8004951s CPU1, avec les544
mises à jour prévues et aucune erreur. Newton, candidat primaire, satisfait
les trois clauses fixées. Adam, témoin annoncé, les satisfait aussi, avec une
détection incomplète au seuil0,5. Ce résultat ne démontre pas la maîtrise d’un
jeu ni l’accélération de l’apprentissage du suivant.

## Ce qui a été comparé

Les deux recettes utilisent le même modèle : six coefficients RGB partagés
entre tous les pixels, plus un biais, en float64. Tous les coefficients partent
àzéro. Aucun poids précédent, règle de couleur, coordonnée ou identité de cas
n’est fourni à l’apprenant. Les cibles restent les différences des branches
déjà observées dans la construction stationnaire certifiée.

Newton amorti7/4 dispose de16mises à jour par bras ; Adam3e-4/clip1 en dispose
de256. Chacun possède un bras visuel et un bras à entrée constante. Les quatre
bras alternent selon l’ordre déclaré. Ce sont deux recettes à budgets distincts,
pas une comparaison à calcul égal ni l’effet causal isolé d’un optimiseur.

| Mesure finale | Newton visuel | Adam visuel |
|---|---:|---:|
| IoU moyenne par paire, puis entre paires non vides | 1,000000 | 0,565774 |
| Avance sur le meilleur de ses trois témoins | 0,997479 | 0,563253 |
| AP moyenne | 1,000000 | 1,000000 |
| Pixels positifs retrouvés sur847 | 847 | 546 |
| Faux positifs sur tous les cas | 0 | 0 |
| Faux pixels des deux cas vides | 0 / 0 | 0 / 0 |
| BCE équilibrée | 0,217739 | 0,632429 |
| Critère IoU≥0,25, avance≥0,10, chaque videFP≤0,001 | PASS primaire | PASS témoin |

L’initial commun prédit tout positif : IoU0,00252083 et100% de faux positifs
sur les deux vides. Les deux constantes apprises et la sortie zéro ont une
IoU nulle. Les BCE des constantes Newton/Adam valent0,679401/0,687736.

## Les douze cas, sans sélection

Newton retrouve tous les positifs et ne produit aucun faux positif dans aucun
cas. Adam retrouve respectivement0,2,10,41,75,166,0,0,221,24,6,1pixels, pour
des cibles de1,4,16,64,128,256,0,0,336,32,8,2pixels. Son PASS moyen n’implique
donc pas la réussite du cas àun pixel, dont il manque l’unique positif.

Son AP1 signifie que les positifs sont classés avant les négatifs, dans chacun
des dix cas non vides. Certains scores positifs restent pourtant sous0,5.
Le seuil ne change pas ; aucun balayage après résultat. Newton obtient à la
fois ce classement et les masques exacts sur le lot.

Les compteurs séparent1093produits linéaires et1093objectifs,544gradients,
32Hessiennes,32résolutions,512propositionsAdam et544retours de mise à jour.
Aucun nouveau tirage, frame d’émulation, gradient ou entraînement caché dans
l’évaluation finale. Les poids initiaux, quatre états finaux, cinq jeux de
logits/probabilités et le journal entier sont conservés.

## Interprétation et suites

Une recette de premier ordre suffit ici à atteindre le critère dans une
paramétrisation simple. Cela ne réhabilite pas le CNN : son paramétrage,
initialisation et précision sont différents, et son échec reste archivé.
Le résultat affine n’identifie pas une cause unique de cet échec.

La vérification algébrique séparée, déclarée dans
`ARC4_STATIONARY_AFFINE_FAMILY_DRAFT.md`, est maintenant complète et revue.
Elle certifie la fonction affine Newton sur toute la famille stationnaire
admissible : logit positif minimal1,315005504 et logit négatif maximal−0,514252547.
Les extrema sont calculés en rationnels exacts à partir des coefficients appris.
Cela couvre tous les arrangements de pixels et bruits de couleurs autorisés
dans ce domaine, sans nouveaux pixels ni fit, mais pas tous les arrondis de
noyaux numériques ni les images hors de cette famille.

Adam n'obtient pas ce certificat : son minimum positif vaut−0,049458532.
L'initial et les deux constantes ne le satisfont pas non plus. Les cinq résultats
sont conservés, sans remplacer le candidat primaire ni changer le seuil.
Seules les sommes des coefficients temporels sont identifiables ici : les deux
passés avant intervention sont identiques, donc aucune compréhension du mouvement
ne découle de cette preuve.

Jugement3133ac0 :21,9843s, deux archives/24tableaux validés, cinq calculs,
43053octets, aucun produit de modèle, gradient ou entraînement. Revue indépendante
18613c4 :1,7314s, cinq recalculs rationnels supplémentaires et distincts,117sources,
10preuves et24signatures vérifiées depuis les JSON, sans rouvrir les NPZ.
Résultat SHA129fd18914406329ee872444b8ffe573416e3922fd85f1e21f1f47ed351d5711 ;
revue SHA25e591a48725d2b744049ec1c469fae0b9da403f4d48677655bd86cafe836dc1.
L’ancienne validation conditionnelle775 du CNN demeure fermée.

## Pièces et état de la revue

Sources mathématiques63e541c/70e874c, instrument/protocole d39e496,
préflight indépendant110a599 :112sources et5preuves, aucune donnée décodée.
Résultats3ef0cf9 dans `craft_v6_out/v70_stationary_affine_fit/`.
`fit_v0.json` SHA256
`c60b92fea8846af3cbc8bfe9db95bdfc1585ad9c661b276462e9db67f2aa8014`.

Revue indépendante conforme, vérificateur4366891 committé avant lecture
scientifique :112sources,25preuves,8245événements et tous les compteurs vérifiés.
Les métriques ont été recalculées depuis les logits sauvés, sans produit de
modèle, gradient, optimisation ou émulation. Durée26,62s, continuité des pertes
exacte et aucune hausse de BCE Newton.
`fit_review_v0.json` SHA256
`8be644481304ed9393f62fa93232ed5534dc46f5ee5fa2775226b710f0a646f5`.

Figure lisible `craft_v6_out/v70_stationary_affine_fit/fit_v1.png`, commit4b79801,
générée depuis les JSON revus uniquement et inspectée. Le premier renduv0 est
conservé ;v1sépare la légende d'un libellé d'axe, sans changer les données.
