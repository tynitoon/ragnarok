# Calibration linéaire Atari : résultat négatif

10 septembre 2026. La tentative unique est complète en **4,6822162 secondes**
sur un CPU, avec les 32 mises à jour prévues et aucune erreur technique.
Les résultats sont scellés b08e0d2 ; la revue indépendante b64bae7 est conforme.
Le modèle visuel final prédit partout négatif au seuil fixé de 0,5.

## Recette et population

Sept coefficients float64, initialisés à zéro, utilisent les six intensités
RGB locales des deux images passées et un biais. Newton amorti 7/4 effectue
16 étapes visuelles et 16 sur entrée constante, dans l’ordre annoncé. Aucun
poids du contrôle artificiel n’est chargé. Les images et labels sont les
douze ancres Atari déjà observées et revues ; aucun nouveau tirage, frame,
jeu réservé ou passage GPU n’a eu lieu.

Cette recette a été choisie après les résultats précédents sur ce lot de
développement. Le protocole déclaré c3300df le dit explicitement ; ce n’est
pas une confirmation indépendante. Les fichiers de référence sont
`docs/ARC4_ATARI_AFFINE_DRAFT.md` et `ARC4_ATARI_AFFINE_PROTOCOL.json`.

## Résultat au seuil annoncé

| Mesure | Visuel appris | Constant appris | Initial | Critère visuel |
|---|---:|---:|---:|---|
| IoU moyenne non vide | 0 | 0 | 0,004902 | ≥0,25 : FAIL |
| Avance sur le meilleur témoin | −0,004902 | — | — | ≥0,10 : FAIL |
| Faux pixels, Seaquest ancre0 | 0 | 0 | 33600 | ≤33 : PASS |
| Faux pixels, Seaquest ancre1 | 0 | 0 | 33600 | ≤33 : PASS |
| AP moyenne non vide | 0,098645 | 0,004902 | 0,004902 | Descriptif |
| BCE équilibrée finale | 0,669718 | 0,679401 | 0,693147 | Descriptif |

Le témoin de sortie nulle a une IoU nulle, une AP de 0,004902 et aucun faux
positif. L’amélioration du classement mesurée par l’AP ne produit aucun
pixel positif à la décision finale. La bonne spécificité sur les deux vides
ne compense pas l’absence de détection des 1647 pixels cibles.

| Jeu | Ancre | Pixels cible | Vrais positifs | Faux positifs | IoU | AP |
|---|---:|---:|---:|---:|---:|---:|
| Pong | 0 | 144 | 0 | 0 | 0 | 0,004240 |
| Pong | 1 | 296 | 0 | 0 | 0 | 0,007424 |
| Breakout | 0 | 344 | 0 | 0 | 0 | 0,013545 |
| Breakout | 1 | 184 | 0 | 0 | 0 | 0,010586 |
| Freeway | 0 | 127 | 0 | 0 | 0 | 0,019974 |
| Freeway | 1 | 127 | 0 | 0 | 0 | 0,019579 |
| Seaquest | 0 | 0 | 0 | 0 | — | — |
| Seaquest | 1 | 0 | 0 | 0 | — | — |
| Ms Pac-Man | 0 | 109 | 0 | 0 | 0 | 0,167130 |
| Ms Pac-Man | 1 | 184 | 0 | 0 | 0 | 0,102386 |
| Asteroids | 0 | 78 | 0 | 0 | 0 | 0,233194 |
| Asteroids | 1 | 54 | 0 | 0 | 0 | 0,408388 |

Les moyennes sont calculées par ancre, puis entre les cinq jeux non vides.
Les deux Seaquest restent des conditions séparées. Leur absence de réponse
sur ces huit images ne signifie pas que Seaquest est incontrôlable.

![Les douze ancres et les témoins](F:/dev/ragnarok/craft_v6_out/v70_atari_affine_fit/fit_v0.png)

La figure a été produite une fois à partir des JSON et du journal, sans
tableaux ou inférence supplémentaires, puis inspectée sans défaut de rendu.

## Interprétation et limites

Cette recette bornée n’atteint pas le critère sur ces enveloppes Atari.
La méthode qui avait appris l’association artificielle ne reproduit pas ce
succès sur ce lot réel. Cela ne prouve ni la nécessité du contexte visuel,
ni une impossibilité de séparation linéaire : 16 étapes ne certifient pas
l’optimum, et l’optimum BCE ne garantirait pas toutes les conditions IoU/FP.

La comparaison avec les CNN précédents change architecture, initialisation,
précision et optimiseur. Elle n’isole pas l’effet causal d’une composante.
Les labels représentent des coordonnées de réponse observée à des
interventions, pas des objets identifiés ni une preuve de contrôle direct.
Aucun apprentissage de jeu, transfert ou accélération n’est démontré.

Ce résultat clôt la série de calibrations linéaires locales annoncée.
Aucun seuil, nombre d’étapes ou taux n’est ajusté pour obtenir un succès.
Les anciennes validations conditionnelles restent non exécutées. Le prochain
jalon concerne la distinction entre réponse RGB observée et attribution à
une instance, avec une admissibilité inconnue tant que la correspondance
n’est pas établie.

## Provenance et conservation

Le préflight v0, arrêté sur l’ancien schéma de résultat avant tout décodage,
est conservé en 2a8d09e. La correction explicite 511ce8d ne change pas la
recette. Les 15 fixtures du lanceur passent en 5,53 s ; trois témoins de
schéma supplémentaires passent en 2,90 s, sans répéter les quinze premiers.

Le préflight v1 deecfd3 vérifie 96 sources, neuf preuves et le runtime exact,
en 11,8121375 s cumulées, arrêt v0 compris, sous sa limite de 60 s. Son SHA :
31d7a04dae6cdef365a3cc4aeee51db0adb4cfc4660375fab612d2066cecf23b.

Le fit a effectué exactement 67 produits linéaires, 67 objectifs, 32 gradients,
32 Hessiennes, 32 résolutions et 32 mises à jour. Les états initiaux et finaux,
les trois sorties de logits et probabilités, le journal et les reçus sont
préservés. Résultat SHA :
b168939f7fb3c4fcc81684785de64b49d79a24a07f4500b56e76d6437ac1f121.

La revue indépendante vérifie 96 sources, 30 preuves, les 557 événements,
48 lignes de métriques et trois BCE en 21,9469968 s sur un CPU. Les pertes
avant/après sont continues et aucune étape Newton n’augmente l’objectif.
Aucun produit de modèle, gradient, résolution ou mise à jour n’est répété
pendant cette revue. Son reçu SHA :
b6c2a748f962e42a77717b846f4d23b92f9973d43599bb5a2f1b6a0a7e8bc732.
