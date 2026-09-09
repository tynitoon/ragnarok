# ARC 4 — revue scientifique du témoin à champs libres

10 septembre 2026. Seconde lecture scientifique du résultat conservé dans
`craft_v6_out/v70_free_displacement_fit/fit_v0.json`, SHA-256
`c4e72945902e06430e2cff84bf604d9860f6aaef110bba2a12fd4775f41b9e68`.

Cette revue utilise uniquement les documents, le code déjà consulté et de
l’arithmétique sur les tableaux JSON existants. Aucun modèle, optimiseur,
émulateur, benchmark ou test n’a été exécuté. Elle ne certifie pas la fermeture
des preuves ou les comptes, examinés séparément. Les anciens essais, sources,
budgets et critères restent inchangés.

## 1. Verdict conservé et portée du témoin

Le témoin optimise directement deux champs indépendants de 16 × 16 pixels,
soit 1 024 paramètres initialisés à zéro. Le protocole déclare 256 mises à jour
Adam, un pas de 0,05 dans les coordonnées de déplacement, et la perte RGB aux
largeurs 1, 3 et 7. Cette capacité par scène ne constitue pas une fonction apprise
capable de traiter de nouvelles observations.

La perte native finale est **0,013014632277190685**, supérieure au seuil strict
**0,01171875**. Le témoin est donc **négatif**. La réussite d’une seule scène ne
valide pas le critère global. Ce résultat n’établit ni une impossibilité
d’ajustement, ni perception, généralisation, compétence Atari, transfert ou
apprentissage accéléré.

| Observation finale | Scène 0 : cible droite | Scène 1 : cible gauche |
| --- | ---: | ---: |
| Erreur native enregistrée | 0 | 0,02602926455438137 |
| Somme RGB de la source et de la cible | 18 | 18 |
| Somme RGB reconstruite sur la cible | 18 | 0 |
| Somme RGB reconstruite hors cible, recalculée | 0 | 1,9904750688001513 |
| Pixels de reconstruction non nuls | 6 | 4 |
| Pixels sans couverture sur 256 | 7 | 6 |

Les sommes recalculées accumulent les valeurs JSON en précision double. Leur
écart de dernier ordre avec les sommes float32 enregistrées ne change aucun
verdict : la somme RGB enregistrée de la scène 1 est 1,9904751777648926.

## 2. La scène gauche montre une contraction et une dilution intérieures

Les six pixels blancs sources sont aux colonnes 4 à 6, lignes 6 et 7. Dans le
champ final de la scène 1, leurs destinations sont toutes comprises entre :

```text
x = 4,9359115362 à 4,9964975726
y = 6,9719113111 à 6,9985405803
```

Ils se concentrent donc près de `(5, 7)`, dans l’écran et près de leur ancienne
position. Le splatting ne les envoie pas sur la cible gauche, située aux
colonnes 0 à 2. Leur contribution lumineuse reste limitée aux quatre positions
`(4,6)`, `(5,6)`, `(4,7)`, `(5,7)`, où la cible est noire.

À partir des couvertures et reconstructions conservées, pour un canal RGB :

```text
somme(couverture × reconstruction) = 6,0000000301
somme(couverture aux quatre pixels lumineux) = 14,0091437995
contribution de couverture noire correspondante ≈ 8,0091437694
```

Le premier calcul retrouve, aux arrondis float32 près, les six unités de poids
blanc sources. Elles n’ont donc pas disparu hors écran. Leur concentration,
puis la moyenne avec les contributions noires, réduisent leur somme RGB de
18 à environ 1,990475, soit une diminution de **88,941805 %**. La couverture
atteint **11,5446624756** au pixel `(5,7)`.

La perte native se décompose en la cible entièrement manquée et le résidu gris
hors cible :

```text
L1 scène 1 = (18 + 1,9904750688) / (3 × 16 × 16)
           ≈ 0,0260292644125
```

Cette valeur rejoint la mesure enregistrée à l’arrondi près. Le résidu gris
rend même cette image moins bonne que le témoin entièrement noir de perte
0,0234375. La concentration et la dilution sont ici des faits observés dans
les tableaux ; elles ne sont plus seulement des mécanismes possibles.

## 3. L’image droite exacte ne valide pas un mouvement physique

Dans la scène 0, les six pixels blancs sont bien reconstruits aux positions
cibles et aucun autre pixel ne reste lumineux. Toutefois les déplacements
des six pixels blancs sources ne sont pas une translation uniforme :

```text
dx = 2,5762593746 à 5,3340415955
dy = −0,6617524624 à 0,9774141908
```

La couverture sur les six pixels cibles varie de **0,1494237483** à
**2,3902976513**. Sa somme vaut environ 6 ; il n’y a pas de contribution noire
à ces six destinations, puisque la reconstruction y vaut exactement 1 et
`couverture × reconstruction` égale la couverture.

La moyenne de contributions blanches produit donc une image exacte malgré un
champ déformé. Cela démontre l’insuffisance de l’exactitude RGB pour identifier
la géométrie du déplacement. Les pixels blancs sont identiques et le fond noir
est sans texture : leur correspondance physique n’est pas observable de façon
unique depuis ces seuls écrans.

Le scénario du juge précise un rectangle opaque en translation sur un fond
immobile. Les mots « physique » et « fond » se rapportent à cette déclaration,
pas à une identité révélée à l’agent par les observations RGB.

## 4. Des cibles de premier plan erronées sont déductibles du champ final

La scène 0 contient six pixels initialement noirs aux positions où arrive le
rectangle blanc : colonnes 8 à 10, lignes 6 et 7. Le champ final les déplace
d’environ deux pixels horizontalement et deux pixels verticalement vers des
destinations noires intérieures. Tous dépassent le seuil de mouvement 0,25.

Pour chacun, l’image future vaut 1 à la position fixe et son échantillonnage
bilinéaire à la position prédite vaut 0. La formule de
`motion_foreground_target()` donne donc :

```text
erreur stationnaire = 1
erreur après déplacement = 0
cible m = (1 − 0) / (1 + 1/255) = 255/256 = 0,99609375
```

Ces pixels appartiennent pourtant au fond du scénario déclaré. Leur déplacement
artificiel évite la dilution du blanc arrivant et explique leur couleur mieux
que rester à une position ensuite occultée.

Il s’agit d’une **cible m déduite arithmétiquement** des images, déplacements et
de la formule, et non d’une tête m entraînée, exécutée ou mesurée. Ce témoin
n’a aucun réseau. Le calcul identifie néanmoins une ambiguïté concrète de
l’auto-supervision P → m : un bon score d’image peut fournir du premier plan
erroné. Dans la scène 1, les six pixels noirs aux positions de la cible gauche
ont, eux, un déplacement exactement nul.

Les cibles C utilisent les instances et les projections prédites ; elles ne
peuvent être déclarées correctes sur la seule base d’un bon score P. Aucun
résultat C ou ME n’est mesuré dans ces tableaux. La dépendance de démarrage
entre instances, C, ME et adaptateur d’effets reste ouverte.

## 5. Le témoin constructif établit une limite structurelle d’occlusion

Les sources du témoin constructif committé dans
`d4cb32523b33275cfb6eda1238fe5e4ae74277fb` ont été lues, sans exécution :
`docs/ARC4_PREDICTION_OCCLUSION_LIMITS.md` et
`tests/test_prediction_occlusion_limits_v70.py`.

Il déclare un rectangle blanc opaque 3 × 2 sur un fond noir immobile. Cette
interprétation physique appartient uniquement au juge. Les trous noirs laissés
à l’ancienne position sont compatibles avec le fond nouvellement découvert
dans ce cas particulier.

Déplacer seulement les six pixels blancs de quatre pixels vers la gauche ou
la droite amène chaque blanc sur un noir immobile. La couverture vaut 2 et la
moyenne donne du gris 0,5, au lieu du blanc opaque attendu. Ainsi :

```text
L1 = 6 × 3 × 0,5 / 768 = 0,01171875
```

Le mouvement déclaré correct atteint exactement la limite exclue par le
critère strict. Le rectangle poolé a les mêmes dimensions dans la source et
le futur ; ce pooling n’est pas la cause de cette dilution. Celle-ci vient de
la moyenne des collisions sans traitement de l’occlusion.

La seconde construction déplace également les six pixels noirs de destination
de quatre pixels vers le bas, dans du noir. Elle obtient une image exacte mais
donne une cible de premier plan élevée à ces six pixels noirs. Cette
construction est indépendante des champs optimisés, dont les déplacements
noirs finaux ne sont pas les mêmes.

### Borne de Jensen avec fond immobile

La limite ne dépend pas du déplacement précis des blancs. À chacune des six
destinations, le fond immobile apporte une couverture 1. Si les blancs y
apportent un poids `w_i`, la luminosité est `w_i / (1 + w_i)`.

Chaque pixel blanc source apporte au plus une unité de poids dans l’écran,
donc `somme(w_i) <= 6`. La fonction `w / (1 + w)` est croissante et concave
pour `w >= 0`. Jensen donne alors :

```text
somme_i w_i / (1 + w_i)
  <= 6 × [(somme_i w_i)/6] / [1 + (somme_i w_i)/6]
  <= 3

L1 sur les seules destinations >= (6 − 3) / 256 = 0,01171875
```

Les erreurs ailleurs sont positives ou nulles. Aucune redistribution des seuls
blancs ne peut donc passer le seuil strict tout en laissant ce fond immobile.
C’est une borne sur le reconstructeur et le scénario déclarés, pas une mesure
supplémentaire ni une preuve que tous les jeux ou reconstructeurs échoueraient.

## 6. Causes ouvertes et une décision de conception

La moyenne des collisions est incompatible avec l’occlusion opaque du scénario
déclaré : ce défaut structurel est distinct de la question de convergence. Les
tableaux établissent aussi une contraction et une dilution dans la scène gauche,
mais ne déterminent pas pourquoi l’optimisation les a privilégiées.

L’initialisation nulle, Adam et son budget, les gradients du splatting aux
coordonnées entières, ainsi que la proximité du bord gauche et le filtrage à
bordure nulle restent des causes possibles de l’asymétrie. Aucune n’a été
isolée par ce témoin. Un seul essai ne permet pas d’attribuer l’échec à l’une
d’elles, ni de justifier un nouveau pas, une graine ou une prolongation.

La décision recommandée est de **définir un contrat distinct de reconstruction
P qui traite la visibilité et l’occlusion**, avant de qualifier ses déplacements
de supervision fiable pour m/C. Ce contrat doit pouvoir expliquer le rectangle
qui couvre le fond immobile sans le griser ni forcer le déplacement du fond.
Une éventuelle visibilité apprise devra venir des propres images, sans masque,
identité ou déplacement oracle ; les correspondances occultées ne devront pas
devenir automatiquement une preuve de premier plan ou de contrôle.

Cette décision ne choisit ici ni formule de reconstructeur, ni régularisation,
ni protocole d’apprentissage, et n’autorise aucun nouvel essai. Elle ne résout
pas encore le démarrage ME → effets. Les anciennes mesures et leurs verdicts
négatifs restent conservés.

## 7. Sources réellement utilisées et empreintes

Empreintes SHA-256 relevées lors de la mise en document. Les sources de P, de
la perte floutée, de C, de la vision §2.1–2.2 et des limites d’amorçage avaient
déjà été consultées lors de la première revue ; elles ont servi à interpréter
ces nouveaux tableaux. Aucun résultat antérieur n’est réévalué ici. Aucune
source de littérature n’est invoquée.

| Fichier relatif à la racine du dépôt | SHA-256 |
| --- | --- |
| `docs/ARC4_FREE_DISPLACEMENT_FIT_DRAFT.md` | `fa0045af04ac6a9d38febe75e54141c7c041f725f3187533ea068f935df83477` |
| `craft_v6_out/v70_free_displacement_fit/fit_v0.json` | `c4e72945902e06430e2cff84bf604d9860f6aaef110bba2a12fd4775f41b9e68` |
| `ragnarok/learning/pixel_prediction_v70.py` | `35fe92dd9970925d2a1a69128d6349adf9ce12d6d57c403f7ec371d320fe6c77` |
| `ragnarok/learning/blurred_prediction_loss_v70.py` | `10329c2830b10564bd139e6717fed6ead8166666eb3124801e3d5adff3df4dc4` |
| `ragnarok/learning/counterfactual_targets_v70.py` | `59dd4178a93a06c38334e708c7910a6bc702b796bb38e781dc65a41d2d000e85` |
| `ARC4_VISION.md` | `250f5b65b0d94d311af66bb77418cefc80f5fc68e4bbd93208aeb49320e80450` |
| `docs/ARC4_BOOTSTRAP_LIMITS.md` | `024208107d9ee16f17fe739e2f6bdb37775aecb3f2d560171a8f8bac4d22b51c` |
| `docs/ARC4_PREDICTION_OCCLUSION_LIMITS.md` | `8aa1741fc12e611005bfaeac02f814639497bef4969e0cbe2c469f056303fbc9` |
| `tests/test_prediction_occlusion_limits_v70.py` | `9f4d6b889909492ce04181ce9123e3ba05855ba7c30f6a8bab790b2ae32784b4` |
