# Calibration comparative affine après les deux échecs du CNN

10 septembre 2026. Nouvelle expérience de développement après le fit Atari et
le fit stationnaire négatifs, tous deux conservés. Les critères de ces essais
ne changent pas. Aucun nouveau tirage ni fit n’est exécuté par cette déclaration.
Le contrôle stationnaire possède une solution connue du CNN ; sa recette Adam
bornée ne l’a pas atteinte. Il faut maintenant vérifier un apprentissage effectif
sur une paramétrisation plus simple, sans attribuer d’avance l’échec à la profondeur.

## Comparaison principale et témoin

Une régression logistique affine, sept paramètres partagés sur tous les pixels :
six coefficients pour les deux RGB ordonnés divisés par255 et un biais.
Aucune coordonnée, identité de cas, loi de couleur, commande, poids du certificat
ou poids antérieur en entrée ou en initialisation. Les sept coefficients partent
àzéro, de façon générique, dans tous les bras. Les passés et les labels issus
des branches sont les seules données décodées de l’archive certifiée774.

Deux recettes sont comparées sur ce MÊME modèle et la MÊME précision float64 :

- candidat primaire :16 mises à jour Newton amorti, pas entier, damping7/4 ;
- témoin d’optimisation :256 mises à jour Adam3e-4, betas0,9/0,999, eps1e-8,
  clip de norme1, sans weight decay.

Chaque recette a son propre bras à entrée visuelle et son bras à entrée nulle,
initialisés identiquement. Pour k=1…256 : Newton visuel puis constant si k≤16,
puis Adam visuel et constant. Total544 appels de mise à jour prévus. Un pas
exactement nul à l’optimum du témoin constant reste une itération retournée ;
il ne sera pas présenté comme une amélioration. Aucun arrêt sur un score.

La BCE reste la moyenne par ancre, puis équilibrée entre classes lorsqu’elles
sont toutes deux présentes ; les vides gardent leur masse entière. Les poids
par pixel somment àun ; la masse positive totale vaut5/12. L’initial àlogit0
prédit tout positif au seuil `>=0,5`. Le témoin de sortie nulle est distinct.

## Calcul numérique fixé

Avec le biais, chaque vecteur d’entrée a une norme au carré≤7. Le Hessien H de
la BCE est positif semi-défini et borné par(7/4)I. Les deux images identiques
créent des colonnes redondantes ; aucune inversion directe de H ne sera faite.
Résoudre `(H + 7/4 I) delta = -gradient` et appliquer le pas entier. Le damping
est uniquement dans le système linéaire : ni pénalité ni terme ajouté au gradient.
La formule de descente mathématique est
`F(w+delta) <= F(w) - delta.T H delta - 7/8 ||delta||²`.
Elle ne garantit pas le franchissement du critère en16 étapes.

Calculs float64, sigmoid et BCE stables. Après chaque retour de mise à jour,
réévaluer la BCE complète. Pour Newton seulement, une hausse supérieure à
`1e-12 * (1 + abs(loss_avant))` déclenche un arrêt technique, sans recherche
d’un autre damping ou pas. Vérifier gradients, Hessien et coefficients finis.
Les appels d’objectif, gradient, Hessien, prédiction et les mises à jour auront
des compteurs distincts d’entrée/retour ; une opération interrompue reste incertaine.

## Verdicts fixés et limites

Évaluation des probabilités initiales communes et des quatre bras finaux
uniquement ; le témoin zéro ne nécessite aucun produit du modèle. Pour chaque
recette, publier le même critère : IoU moyenne≥0,25, avance≥0,10 sur le meilleur
de son constant appris, de l’initial commun et de la sortie zéro, FP≤0,001 pour
chacun des deux vides. Même ordre et pondération par paire, douze lignes entières,
AP avec scores ex aequo groupés. Les métriques ne choisissent pas un solveur :
le verdict primaire est celui de Newton, Adam reste un témoin annoncé.

Les deux PASS montreraient qu’une recette de premier ordre suffit dans cette
paramétrisation simple. Newton PASS et Adam FAIL montreraient qu’un échec de
cette recette Adam existe même sans profondeur, et ne justifieraient pas à eux
seuls un raccord résiduel au CNN. Newton FAIL impose l’examen du solveur avant
un tel raccord. Aucune issue ne désigne une cause unique de l’échec CNN : la
paramétrisation, l’initialisation et la précision diffèrent de cet ancien fit.

Une tentative CPU1, CUDA vide,300s dont10s de clôture,64MB de sorties binaires.
Imports/preuves initiales hors temps ; données, calculs, coefficients et toutes
les sorties scientifiques, fsync/hash, journal et provenance finale inclus.
Seul le petit reçu final et la console suivent la mesure. Zéro nouvelle image,
génération ou émulation ; aucun retry, changement de seuil ou prolongement.
Sources, protocole, tests mathématiques et de panne, revue et préflight doivent
être committés avant exécution. Une erreur ou un dépassement donne INCOMPLET.

Un PASS établirait l’apprentissage d’une association visible sur ce lot. Il
ne réhabiliterait pas le CNN, ne ferait pas passer l’ancienne validation
conditionnelle775, et ne démontrerait ni perception générale ni jeu maîtrisé
ni représentation réutilisable ni accélération par expérience antérieure.
Toute vérification de nouvelles entrées ou de garanties sur toute la famille
devra être déclarée séparément, avec les poids gelés et exclusivement dans le juge.
