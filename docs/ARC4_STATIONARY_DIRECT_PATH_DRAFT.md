# Candidat suivant : une voie RGB directe dans le CNN

10 septembre2026. Proposition prospective après les résultats négatifs du
CNN et le résultat affine positif revu. Aucun nouveau fit, nouveau tirage ou
changement du CNN archivé n’est exécuté par ce document.

## Hypothèse testable

Le CNN peut représenter la règle simple, mais sa recette Adam bornée n’y
parvient pas. Le modèle affine obtient un PASS avec Adam dans une autre
paramétrisation. Cela justifie d’examiner une voie affine directement reliée
aux logits, sans conclure que la profondeur ou l’optimiseur est l’unique cause.

Le candidat conserve exactement le CNN de10177paramètres et lui ajoute une
convolution1×1 des six canaux RGB vers un logit, avec biais : sept paramètres
supplémentaires, tous initialisés àzéro. Sa sortie est la somme du logit CNN
et de cette voie directe. Les10184paramètres sont entraînés ensemble. Aucun
coefficient affine appris, poids du juge ou règle de couleur ne sera transféré.
Le modèle ne contient ni coordonnées ni paramètre lié à une palette ou un jeu.

Construire d’abord le CNN avec le seed764 et l’ordre de tirages inchangés,
puis les sept zéros sans consommer de hasard. Les coefficients CNN initiaux
doivent correspondre exactement à l’archive de son ancien fit. La voie nulle
conserve sa fonction initiale. Le bras à entrée constante en reçoit une copie.
Précisionfloat32, mêmes douze cas observés, même BCE équilibrée et même Adam
3e-4/clip1,256mises à jour par bras. C’est une modification d’architecture
annoncée, avec le témoin CNN antérieur archivé sous la même recette ; aucune
réexécution de ce témoin n’est prévue.

## Déclaration de résultat

Critère de développement inchangé : IoU≥0,25, avance≥0,10 sur le meilleur
de la constante apprise, de l’initial et de la sortie zéro, FP≤0,001 sur chacun
des deux cas vides. Toutes les douze lignes et les trajectoires de BCE sont
publiées. Ni arrêt anticipé sur un score, ni changement de seuil, ni allongement
des256étapes. Les anciennes validations conditionnelles restent fermées.

Une tentative CPU1, sans GPU,1200s dont10s de clôture et64MB de sorties.
512updates et trois évaluations de lot : initial commun et deux bras finaux.
Comme auparavant, compter toutes les entrées/retours de modèle et mises à jour,
préserver les logits uniques avant une erreur de métrique, les paramètres
partiellement mis à jour et toute incertitude. Tout calcul, conservation des
sorties et clôture scientifique doivent tenir dans la même limite.

Un PASS montrerait qu’un ajout générique de sept paramètres rend cette recette
suffisante sur ce lot. Il ne désigne pas pourquoi l’ancien CNN a échoué, ne
prouve pas que la voie directe porte à elle seule la décision et ne prouve pas
la réussite sur les cibles Atari. Un FAIL conserve la borne de cette recette
sur ce candidat ; pas de nouvel essai de taux ou de profondeur dans ce protocole.

Sources, protocole détaillé, vérifications de l’initialisation/voie directe,
comptabilité et revue contradictoire sont requis avant le lancement. La suite
conditionnelle éventuelle doit être déclarée séparément ; ce document ne
permet ni d’ouvrir les jeux réservés ni de lancer le gate0 non gelé.
