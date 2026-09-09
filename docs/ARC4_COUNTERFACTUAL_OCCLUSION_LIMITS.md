# C : une position projetée dans l’image ne certifie pas une correspondance visible

Témoins externes du 10 septembre 2026. Le nouveau fichier
`tests/test_counterfactual_occlusion_limits_v70.py` utilise le helper gelé
`component_targets` sans le modifier. Les anciens protocoles, résultats,
budgets et critères restent inchangés.

Les identités A et B, les états physiques, leur réponse à l’intervention et
l’ordre des objets opaques appartiennent exclusivement au juge synthétique.
Aucun modèle n’est exécuté ou entraîné. Le helper reçoit seulement son API
habituelle : instances synthétiques, champ synthétique, deux `RGBBranchTrace`,
horizon et choix RGB brut/poolé. Aucun argument d’identité, de mouvement vrai,
d’occlusion ou de visibilité ne lui est ajouté.

## Construction commune et portée du témoin

Chaque écran fait 10 × 12 pixels RGB. À l’ancre commune, le sujet A est le
pixel blanc situé en `(x=2, y=3)`. Une hypothèse de perception fixe fournit
un composant à cette position et un déplacement de quatre pixels vers la droite,
soit une position prédite `(6,3)` à l’horizon de quatre images. L’instance est
produite par `extract_instances` à partir de sorties denses synthétiques.

C’est une facilité accordée pour isoler le calcul de C, pas une perception
apprise. Ces hypothèses sont identiques entre les mondes comparés. Dans chaque
monde, la position prédite est aussi la position physique de A au quatrième
écran de l’intervention : le résultat ne repose donc pas sur une erreur de
position prédite dans cette branche.

Les traces utilisent le vrai type `RGBBranchTrace`, avec les RGB bruts, le
maximum exact des deux images adjacentes et l’endpoint poolé correspondant.
L’ancre est commune et les deux branches contiennent quatre images complètes.
Ce sont des données synthétiques, pas des traces d’émulation payées ni une
preuve du raccord propriétaire en conditions réelles.

## Témoin 1 : différence due à l’objet qui recouvre A

Dans le premier monde, A se déplace de façon autonome aux positions
`x=3,4,5,6`, identiquement dans les deux branches. B se trouve en `(6,3)` et
passe devant A. B reste rouge dans la branche témoin ; l’intervention le rend
vert au quatrième écran. A demeure blanc et son état physique ne change pas
entre les branches. À l’horizon, la différence observée à la position prédite
de A appartient donc à B, qui le recouvre.

Le helper renvoie pourtant pour le pixel source de A :

```text
target = 1
weight = 1
eligible = True
projected_source_pixels = 1
reason = own_motion_and_complete_observation
```

C’est un faux positif si cette cible est interprétée comme la contrôlabilité
physique de A. Le calcul de différence RGB n’est pas incorrect : c’est son
attribution à A qui n’est pas justifiée.

Un second monde produit exactement les mêmes écrans, y compris l’ancre et
toutes les images intermédiaires. B y reste rouge ; A passe devant B et devient
rouge dans la branche témoin, vert dans l’intervention au quatrième écran.
L’intervention agit alors sur l’apparence de A, tandis que son déplacement
reste autonome. A est contrôlé dans ce monde, mais pas dans le premier.

Les tests vérifient l’égalité exacte des RGB bruts, poolés et endpoints entre
ces deux mondes, ainsi que la différence de leurs états physiques causaux.
La cible 1 est identique parce que toutes les entrées disponibles au helper
sont identiques. Les RGB ne permettent pas de choisir entre ces deux causes.

## Témoin 2 : réponse contrôlée cachée dans les deux branches

Une bande rouge opaque B couvre les positions `x=3..6`, à la ligne `y=3`.
A commence visible en `(2,3)`, puis passe derrière B.

Dans le monde où A est contrôlé, ses positions sont `x=3,4,5,6` sous
intervention et `x=3,3,3,3` dans la branche témoin. Sa réponse physique est
donc différente, mais entièrement cachée. La position prédite `(6,3)` est
bien dans l’image et correcte pour l’intervention ; le pixel RGB visible
à cette position appartient à B dans les deux branches.

Le helper renvoie alors :

```text
target = 0
weight = 1
eligible = True
projected_source_pixels = 1
reason = own_motion_and_complete_observation
```

Il ne renvoie pas un cas inconnu de poids nul. C’est un faux négatif si cette
cible est interprétée comme une absence de contrôlabilité physique de A :
l’observation ne permet simplement pas de voir sa réponse.

Un monde alternatif où A suit de façon autonome `x=3,4,5,6` dans les deux
branches produit exactement les mêmes observations. La cible 0 lui est aussi
attribuée. Les tests vérifient là encore l’égalité exacte des entrées malgré
des états causaux opposés. Aucune règle fondée uniquement sur ces entrées ne
peut déterminer si A a répondu à l’intervention derrière B.

## Ce qui est établi, et ce qui ne l’est pas

Les deux témoins passent pour `image_kind='raw'` et `'pooled'`. L’ambiguïté
n’est donc pas créée uniquement par la persistance du max-pool, une branche
tronquée, un mouvement prédit absent ou une projection hors écran.

La variable `visible` de `component_targets` désigne une couverture bilinéaire
positive dans les bornes de l’image. Elle n’atteste ni l’identité du contributeur
observé ni la visibilité du composant source. Le helper conserve correctement
certains cas inconnus géométriques ou temporels ; ces témoins montrent que cela
ne couvre pas l’occultation et la correspondance.

Une cible 0 est correcte comme description « aucun changement RGB observé à
cette projection ». Une cible 1 est correcte comme description « changement
RGB observé ». Les interpréter directement comme des propriétés causales de
l’instance est une hypothèse supplémentaire, invalidée dans les mondes déclarés.

Ces contre-exemples ne mesurent pas une fréquence d’erreur dans Atari et ne
prouvent pas que les anciens réseaux ont appris ces états. Ils ne proposent
ni détecteur d’occlusion ni recette de correction. Une priorité ou une couverture
de reconstruction ne résoudrait pas, à elle seule, l’indiscernabilité de ces
causes physiques ; des hypothèses ou informations propres supplémentaires
devraient être déclarées et leur portée vérifiée séparément.

## Vérification et sources

Quatre cas ont été exécutés avec succès : deux témoins × deux types d’image,
**4 passed in 1.52s**. Interpréteur `C:/Python314/python.exe`, Python 3.14.3,
NumPy 2.4.4, SciPy 1.17.1, pytest 9.0.3. Threads numériques bornés à un,
`CUDA_VISIBLE_DEVICES` vide, chargement automatique des plugins pytest et
écriture du cache désactivés. La vérification après les tests confirme
l’absence d’import de Torch et d’ALE. Aucun moteur natif, GPU ou optimiseur
n’est utilisé.

Sources directes et empreintes SHA-256 relevées après cette vérification ;
cette liste ne constitue pas une revue de toutes les dépendances transitives :

| Fichier relatif à la racine du dépôt | SHA-256 |
| --- | --- |
| `ragnarok/learning/counterfactual_targets_v70.py` | `59dd4178a93a06c38334e708c7910a6bc702b796bb38e781dc65a41d2d000e85` |
| `ragnarok/learning/visual_instances_v70.py` | `6ce6738562f23ad50abe44250c04379aaa325742f7122e251bf29afa6ff60e2d` |
| `ragnarok/environments/rgb_branch_trace_v70.py` | `0519622b291d115a76b879cf9fc5b8a66bd7e491a1f81c726d5b130b417656c3` |
| `ragnarok/environments/rgb_trace_difference_v70.py` | `c9971e40d7c14782968f5ac37e3ae0801788b931376a56ff511ec4a94281af7d` |
| `tests/test_counterfactual_occlusion_limits_v70.py` | `d0a9321b09fb21af5f917c4f0c43a6551f388d99d780460a5badaf2e78f5ed74` |
