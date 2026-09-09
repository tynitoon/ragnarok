# ARC 4 — revue scientifique de l’amorçage du mouvement

10 septembre 2026. Revue contradictoire limitée à la lecture des sources et du
résultat du fit synthétique P natif. Aucun modèle, optimiseur, émulateur ou
benchmark n’a été exécuté pour cette revue. Aucun fichier de résultat du fit
flouté en cours n’a été consulté. Son résultat n’est ni connu ni anticipé ici.

Cette note conserve les critères, budgets, règles STOP, sources et résultats
existants. La recommandation finale est conditionnelle ; elle ne constitue ni
un protocole approuvé ni l’autorisation d’une expérience supplémentaire.

## 1. Ce que le fit natif établit

Le résultat `craft_v6_out/v70_synthetic_motion_fit/fit_v0.json` rapporte deux
scènes synthétiques opposées, utilisées à la fois pour l’ajustement et la mesure,
avec 256 mises à jour Adam par architecture et la seule perte d’image P native.
L’exécution numérique est complète : 512 mises à jour CPU, aucun nouvel écran
émulé, aucun GPU et aucun checkpoint entraîné réutilisable conservé.

La perte finale moyenne est de **0,025212198495864868** pour le parser poolé et
de **0,025491314008831978** pour le parser ordonné. Le critère ordonné gelé,
strictement inférieur à **0,01171875**, n’est pas atteint. Ce résultat établit
un défaut d’ajustement dans ce protocole. Il n’établit ni une impossibilité
d’apprendre, ni une absence de généralisation mesurée, ni un résultat Atari.

### Calcul triangulaire : l’ordre est disponible mais peu exploité

Les fonctions `scenes()` et `measure()` du script définissent les deux cibles
RGB et la distance L1 moyenne entre les deux reconstructions. Notons :

- `T1`, `T2` les cibles ; leur distance moyenne est `D = 0,046875` ;
- `R1`, `R2` les reconstructions finales du parser ordonné ; leur distance
  enregistrée est `delta = 0,00010965496039716527` ;
- `E` la moyenne des deux erreurs L1 par scène.

L’inégalité triangulaire donne :

```text
D <= L1(T1, R1) + L1(R1, R2) + L1(R2, T2) = 2 E + delta
E >= (D - delta) / 2
  >= 0,023382672519801417
```

À elle seule, la faible séparation des reconstructions impose donc une erreur
supérieure au seuil. Pour avoir `E < 0,01171875`, il serait nécessaire que :

```text
delta > D - 2 * 0,01171875 = 0,0234375
```

Cette condition nécessaire ne serait pas suffisante : deux reconstructions
différentes peuvent toutes deux être fausses. Le calcul n’implique pas que les
champs de déplacement eux-mêmes soient presque identiques ; des champs
différents peuvent produire des reconstructions semblables. Il ne démontre
pas non plus une incapacité de l’architecture ordonnée à exploiter ses entrées.

## 2. P peut diminuer sans fournir un mouvement utile

**Établi par les sources et les témoins documentés.** Le reconstructeur déplace
les pixels sources par splatting bilinéaire, normalise les collisions par leur
couverture et laisse les trous noirs. Les trous restent dans le dénominateur de
la perte. Pour le témoin du carré, la perte L1 native est localement plate dans
la famille des translations uniformes qui ne rapprochent pas suffisamment les
supports. Déplacer la source entière hors écran donne une image noire de perte
0,0234375, meilleure que l’image immobile de perte 0,046875.

**Hypothèses, non constatées dans le fit.** Une optimisation peut profiter de la
disparition, des collisions ou de la redistribution des pixels avant d’apprendre
leur déplacement utile. Le fit ordonné termine près de la borne d’une prédiction
commune, mais ses métriques ne permettent pas d’identifier le mécanisme exact.
Elles ne prouvent pas une image noire, une fuite hors écran ou un champ dégénéré.
Les témoins de translation n’affirment pas que tous les gradients d’un champ
dense sont nuls.

La candidate floutée compare symétriquement reconstruction et futur aux largeurs
1, 3 et 7, dans la perte uniquement. Le terme natif reste présent. Un gradient
local orienté dans un témoin ne garantit ni la convergence du réseau ni la
qualité de son mouvement ; c’est précisément ce que le fit distinct doit
examiner sous son protocole déjà gelé.

## 3. Le passage de P au premier plan dépend du mouvement acquis

**Établi par le code.** `motion_foreground_target()` construit une cible
détachée : le déplacement doit dépasser 0,25 pixel, sa destination rester dans
l’image, et le déplacement doit expliquer la couleur mieux que l’immobilité.
Un mouvement encore inutile peut donc donner peu ou pas de cible positive.
La perte BCE du premier plan ne transmet pas de gradient à travers le calcul
de cette cible pour corriger le déplacement.

**Hypothèse de démarrage.** Si cette BCE est activée alors que P n’a pas encore
appris un mouvement exploitable, elle pourrait favoriser un premier plan vide
et réduire ensuite les instances disponibles. Cela dépend du calendrier et
des coefficients d’un apprentissage conjoint ; le fit P image seul ne l’a pas
testé. Les têtes partagent des caractéristiques, ce qui autorise des effets
indirects entre pertes, sans créer cette supervision directe du déplacement.

La moyenne finale de cible de premier plan ordonnée,
`0,010071822442114353`, n’est ni un rappel ni une précision. Elle ne mesure pas
un premier plan appris : ce fit optimise seulement la perte d’image et
n’entraîne pas directement les paramètres de la tête de premier plan.

## 4. C, ME et les effets ne sont pas amorcés par la seule réussite de P

**Établi par le code.** Les cibles C requièrent des instances propres, un
mouvement prédit suffisant, une projection visible et un horizon commun complet.
Sinon leur poids est nul. Une différence visible entre branches ne suffit donc
pas à fournir une cible exploitable. Les cibles numériques détachées ne
transmettent aucun gradient correcteur à travers la projection ; un effet
indirect via les caractéristiques partagées reste possible.

**Limite d’information.** Avec la même paire d’images et le même vecteur d’effet
UNKNOWN, un parser déterministe produit la même prédiction, même si les commandes
enregistrées diffèrent. Les boutons ne sont pas une entrée du parser actuel.
L’ordre temporel peut aider à apprendre un mouvement déjà visible ; il ne crée
pas l’information manquante sur une commande dont l’effet reste inconnu.

**Hypothèse d’architecture encore ouverte.** La vision attend des objets
contrôlés pour identifier ME, puis ME pour construire l’adaptateur des effets,
alors que la prédiction et la projection C utilisent déjà une hypothèse d’effet.
Les raccords numériques ne démontrent pas que cette dépendance peut démarrer.
Une divergence à quatre images ne suffit pas à identifier la divergence au
premier écran exigée pour ME ; la séparation entre contrôle direct, conséquence
indirecte et retard de commande reste à établir. Les absences d’observation et
les préfixes tronqués doivent conserver leur statut inconnu.

## 5. Une seule expérience de recours proposée, sous condition

**Uniquement si le fit flouté échoue**, un témoin d’optimisation directe de deux
champs de déplacement denses libres pourrait aider à localiser la difficulté.
Il conserverait les mêmes images synthétiques, le même reconstructeur et la
même perte floutée, en retirant seulement le réseau de l’optimisation. Les
paramètres des deux champs seraient indépendants : ce serait un instrument
numérique externe, pas une architecture d’agent ni un apprentissage réutilisable.

L’initialisation ne recevrait aucune direction vraie. Seules les images futures
serviraient de supervision ; aucun déplacement vrai, masque, identité ou autre
information oracle ne serait fourni. Avant toute exécution, un nouveau protocole
devrait fixer l’initialisation, le budget, les pas et leur échelle, les mesures
et les règles STOP. Les paramètres d’optimisation du réseau ne sont pas
automatiquement transposables à des déplacements directement paramétrés.

La mesure conserverait le seuil d’image native **0,01171875** et rapporterait
aussi la séparation des reconstructions. Un succès montrerait que ce couple
perte/reconstructeur peut atteindre le seuil avec ces paramètres libres et cette
initialisation ; la paramétrisation du réseau ou son optimisation deviendrait
davantage suspecte. Un échec maintiendrait la perte, le reconstructeur et leur
optimisation parmi les difficultés immédiates, sans démontrer leur impossibilité.

Les paramètres indépendants par scène constituent une facilité supplémentaire.
Même positif, ce témoin ne prouverait ni perception apprise, ni mouvement vrai,
ni généralisation, ni démarrage de C/ME/effets. Cette proposition ne change aucun
critère existant et ne doit pas être interprétée comme un protocole approuvé.

## 6. Sources réellement consultées et empreintes

Les empreintes SHA-256 suivantes ont été relevées lors de la mise en document.
La vision a été lue uniquement aux §2.1–2.2 ; le script de fit a été consulté par
recherche ciblée et pour `scenes()`, `infer()` et `measure()`. Le point de reprise
a servi à orienter la lecture. Aucun autre résultat de fit, aucune archive RGB,
aucun log massif et aucune source de littérature n’ont été consultés. Cette
liste atteste les fichiers utilisés ; ce n’est pas une revue exhaustive de leur
fermeture de dépendances.

| Fichier relatif à la racine du dépôt | SHA-256 |
| --- | --- |
| `docs/CODEX_HANDOFF.md` | `d4e180c95803bb628ea1f1d3d71ba12e5deb22861a0ba186934035817344d6b0` |
| `ARC4_VISION.md` | `250f5b65b0d94d311af66bb77418cefc80f5fc68e4bbd93208aeb49320e80450` |
| `docs/ARC4_BOOTSTRAP_LIMITS.md` | `024208107d9ee16f17fe739e2f6bdb37775aecb3f2d560171a8f8bac4d22b51c` |
| `docs/ARC4_PREDICTION_LOSS_LIMITS.md` | `4c7d514715466bcdfa9f4d5670b94bd61b14fe8baff20dc93c2d138e6a58d8a7` |
| `docs/ARC4_BLURRED_PREDICTION_DRAFT.md` | `435a98302c3e603310eeffccb04c4823a3a89410e16f02b035a3f49f634c6224` |
| `ragnarok/learning/parser_v70.py` | `d18874f74ab1a153b83327e36c48adce8dd699ba0482307893b81dde9eb1a8e9` |
| `ragnarok/learning/ordered_motion_parser_v70.py` | `73a7cfe0b76abc44b260ac7ba9412ec658b65b7024ce2acde6b09b80d47c1daa` |
| `ragnarok/learning/pixel_prediction_v70.py` | `35fe92dd9970925d2a1a69128d6349adf9ce12d6d57c403f7ec371d320fe6c77` |
| `ragnarok/learning/blurred_prediction_loss_v70.py` | `10329c2830b10564bd139e6717fed6ead8166666eb3124801e3d5adff3df4dc4` |
| `ragnarok/learning/counterfactual_targets_v70.py` | `59dd4178a93a06c38334e708c7910a6bc702b796bb38e781dc65a41d2d000e85` |
| `scripts/check_synthetic_motion_fit_v70.py` | `038fdc86f0338e75ab1bb77a496a3e7195494a35c07b5ae42c82bb8fc3f964be` |
| `craft_v6_out/v70_synthetic_motion_fit/fit_v0.json` | `9d049a632816855b5dd2497681c9ff900bdbe04d7d8efbb68bb0cbb56514b82f` |
