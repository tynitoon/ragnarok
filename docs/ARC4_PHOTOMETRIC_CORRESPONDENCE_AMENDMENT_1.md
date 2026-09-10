# Correspondance photométrique — précisions avant implémentation

10 septembre 2026. Cette révision prospective précise et, pour les entrées
poolées, corrige la déclaration f220669. **Aucune donnée, fixture de ce candidat,
initialisation ou mise à jour n’a encore été exécutée.** La déclaration initiale
reste conservée. Les neuf exigences numériques du scorer ne changent pas.

## Temporalité des observations

Le nouveau mécanisme propose des associations entre observations ponctuelles
`raw`. Un `image_kind` explicite provenant du transport propriétaire est requis ;
un tag absent, inconnu, mélangé ou réétiqueté silencieusement est interdit.
Les pixels seuls ne certifient pas ce tag : sa provenance reste à la charge du
propriétaire. Tous les clips procéduraux de l’essai sont raw.

Une séquence fournie seulement en `pooled` donne une proposition inconnue pour
tous les pixels, avec motif `temporal_support_not_point_observation`, **sans
encoder ni calculer d’association**. Cette règle s’applique au modèle initial,
au final et au témoin externe. Les scores non calculés valent `None`, pas une
confiance numérique inventée.

Raison issue de la revue : le max-pooling peut conserver à l’offset 4 un pixel
blanc de l’offset 3 alors que son objet est désormais occulté. Un cycle peut
suivre ce vestige avec une forte cohérence. Refuser le support temporel agrégé
retire cette ambiguïté particulière ; raw ne garantit ni identité ni visibilité.

Les quatre témoins obligatoires restent publiés. Les deux témoins pooled sont
désormais explicitement des **gardes temporelles structurelles**, sans crédit
d’abstention apprise. Les deux raw passent par le mécanisme complet. Les cinq
strates statistiques restent intégralement évaluées par le candidat.

## Lois de scène entièrement précisées

Le rendu calcule exposition × couleur en **float64**, puis applique l’arrondi
au plus proche avec demi-entiers vers l’entier pair (`numpy.rint`) avant conversion
uint8. Les coïncidences dues à cette quantification restent dans leur strate,
sans rejet. Le juge comptera séparément les égalités RGB observées entre l’acteur
interrogé et les autres points ou le fond, sans changer les admissions. Une
identité physique unique ne signifie donc pas une identité observable garantie.

Pour `occluded` et `out_of_frame`, l’acteur 0 en image 1 est tiré uniformément
parmi les 63 cases autres que sa position initiale. Les trois autres points
sont tirés sans remplacement parmi les 63 cases laissées libres.

En image 2 de `occluded`, la case de l’acteur 1 est tirée uniformément parmi
les 64 cases. L’acteur 0 est placé dans cette même case, puis recouvert par
l’acteur 1, rendu après lui. Les acteurs 2 et 3 occupent deux cases tirées
sans remplacement parmi les 63 restantes. Le fond est rendu en premier,
puis les acteurs dans l’ordre 0,1,2,3.

En image 2 de `out_of_frame`, l’acteur 0 est à x=8 et y uniforme dans 0..7,
hors des colonnes 0..7. Il n’est pas rendu. Les positions des acteurs 1,2,3
sont tirées sans remplacement parmi les 64 cases.

Dans `ambiguous`, les deux Bernoulli choisissent chacun une permutation
**relative aux deux cases initiales**, et non par rapport à l’image précédente.
Les acteurs 2 et 3 sont retirés à chaque image suivante, sans remplacement,
parmi les 62 cases hors des deux cases fixes. L’apprentissage tire uniquement
entre `visible_moving` et `visible_stationary`, chacun avec probabilité 1/2.

Pour le témoin RGB, normalisation L2 en float32 avec epsilon 1e−6, identique
à la convention du réseau. Un vecteur noir nul donne un vecteur nul ; aucun
NaN ni direction artificielle. Ce cas existe dans les contrôles historiques.

## Budget, images et inférences

Les 300 s sont un **plafond mural** sur un CPU numérique. Un lanceur standard
pose l’intention et l’échéance avant de créer le processus qui importe Torch.
Les imports et l’initialisation de ce processus sont donc comptés ; leur échec
reste conservé. Le lanceur applique la limite externe, le processus vérifie
coopérativement sa réserve de 15 s, comprise dans les 300 s. Les délais et les
comptes ne sont jamais réinitialisés entre génération, entraînement et inférence.

Le lot de production contient 128+1280 clips, **4 224 images RGB nouvelles**.
Les huit clips historiques de contrôle, 24 vues sélectionnées, sont construits
une fois depuis les définitions de fixtures, conservés puis réutilisés entre
états. Ils ne sont pas des scènes i.i.d. supplémentaires. Zéro émulation.

512 mises à jour proposent 2 048 présentations de clips d’apprentissage, soit
6 144 images encodées. L’évaluation utilise des mini-lots fixes de 16 clips,
soit 80 appels par état pour les 1 280 clips de production. Les quatre témoins
raw sont traités ensemble en un appel supplémentaire par état. Les quatre
pooled sont refusés par le garde, sans appel de modèle.

Attendus pour une exécution complète : **674 appels de l’encodeur appris**
(512 entraînement + 81 final + 81 initial), **13 848 images encodées**.
Le témoin RGB fixe réalise séparément 81 appels d’association sans réseau.
Les 1 288 clips présentés par représentation se répartissent donc en 1 284
associations calculées et quatre refus structurels. Les comptes réels d’appels,
d’images, de matrices, de gradients, de clips et de mises à jour sont conservés.
Les tableaux de scores non calculés des refus pooled ne doivent pas être
remplacés par des résultats numériques.

L’initial est une copie exacte des paramètres avant la première mise à jour,
sans restauration d’un optimiseur ou tampon final. Les propositions des trois
représentations sont scellées avant toute extraction de verdicts. Le juge
conserve les 1 280 verdicts, les huit réponses de branche et leur réduction
aux quatre témoins ; les agrégats seuls ne constituent pas cette preuve.

## Interprétation complémentaire

Un gain de couverture peut venir d’associations déjà correctement classées mais
devenues assez concentrées pour passer la règle d’admission. Le juge publiera
donc aussi, à titre descriptif, l’exactitude du meilleur indice avant abstention
et sa table appariée initial/final sur les requêtes visibles. Aucun nouveau seuil
ou critère n’utilise ce tableau. Même un succès ne suffira pas à attribuer le gain
à une invariance photométrique nouvelle, ni à une représentation d’objet.

Les revues de audit_motion_bootstrap et review_recovery_v62 sont à l’origine
de ces précisions. La modification du domaine temporel a fait l’objet d’une
revue dédiée avant données : favorable, avec déclaration explicite du caractère
structurel des refus pooled. Aucun résultat d’apprentissage n’est anticipé.
