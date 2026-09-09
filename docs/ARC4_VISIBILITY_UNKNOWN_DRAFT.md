# ARC 4 — correspondance inconnue avant les cibles m/C

Décision de conception prospective, 10 septembre 2026. Aucun seuil de sélection
ou entraînement supplémentaire n'est gelé par ce texte. Les anciennes sources,
cibles et mesures restent conservées. Ce document précise ce que le futur
raccord devra établir ; il ne décrit pas un algorithme de visibilité déjà validé.

Le reconstructeur à priorités et sa tête prédisent une image à partir du passé.
Leur bon fonctionnement numérique ne valide pas les anciennes cibles de premier
plan m ou d'effet d'une intervention C. Le témoin opaque établit notamment qu'une
image correcte peut accompagner un déplacement artificiel du fond. La priorité
apprise n'est pas une probabilité calibrée de visibilité.

## Distinguer absence de preuve et cible négative

Toute cible future doit distinguer sa valeur, son admissibilité et la raison
d'une exclusion. Une valeur de stockage nulle avec un poids nul signifie
« inconnu », jamais « immobile » ou « non contrôlable ». L'absence de support,
une projection hors écran, une correspondance occultée ou une observation
manquante ne doivent pas produire automatiquement une cible négative.

Les valeurs et poids construits à partir des observations sont détachés du graphe
de gradient des têtes m/C. Cela empêche un chemin direct qui diminuerait leur perte
en apprenant à exclure les exemples. Cela n'élimine pas la dépendance circulaire :
si le déplacement et la priorité du modèle décident de l'admissibilité, leurs
erreurs peuvent supprimer toutes les données utiles ou retenir des erreurs.
Le nombre d'exemples admissibles et les raisons d'exclusion devront être mesurés.
Une perte nulle avec zéro exemple reste une absence de supervision.

## Ce que couverture et priorité permettent d'affirmer

La couverture géométrique indique que des contributions prédites arrivent dans
une case de l'image. Elle ne prouve ni leur identité ni leur visibilité physique.
La couverture pondérée et les priorités expliquent un mélange du reconstructeur.
Une contribution dominée ou insuffisante peut justifier un veto conservateur,
mais une contribution dominante ne certifie pas une correspondance correcte.
Aucun seuil de dominance n'est choisi dans ce document.

La couverture au point initial, calculée sous les déplacements prédits, ne
décrit pas le monde contrefactuel « ce pixel serait resté sur place ». Une
différence à cet endroit peut venir d'un départ, d'un fond dévoilé ou d'un objet
passé devant lui. La comparer à une destination de même couleur ne tranche pas
ces possibilités, en particulier dans une zone sans texture.

## Conditions nécessaires pour m

L'ancien avantage photométrique compare l'erreur à la position stationnaire et
l'erreur à la destination prédite. Pour le futur raccord, une correspondance
ambiguë ou masquée dans l'une des explications doit conserver le statut inconnu.
Une concordance RGB à la destination ne suffit pas à rendre la comparaison
admissible. Ces conditions sont nécessaires et ne constituent pas un critère
suffisant ou un estimateur disponible.

En particulier, ajouter les priorités au reconstructeur puis appeler l'ancien
`motion_foreground_target` ne répare pas ce problème : cette fonction renvoie
une valeur dense sans poids d'inconnu distinct. Elle reste utilisable seulement
dans le périmètre de ses anciens diagnostics, avec leurs limites déclarées.

## Conditions nécessaires pour C et ME

Une cible C exige les deux branches propres d'une même expérience payée, un
horizon commun complet et une correspondance de l'instance dans chacune des
branches. Le déplacement doit correspondre à l'effet et à l'horizon considérés.
Un support bilinéaire dans l'image n'établit pas ces correspondances. La
disparition ou l'occultation de l'objet dans une branche laisse son effet inconnu.

Une différence à la position projetée peut appartenir à un autre objet. Une
absence de différence ne soutient une cible C nulle que si l'occasion d'observer
l'effet était présente. Même avec une correspondance fiable, une différence
peut être indirecte : elle ne désigne pas automatiquement le personnage contrôlé
ME. Le délai seul ne résout pas cette distinction, comme le montrent les témoins
de propagation et de boutons collants déjà conservés.

## Provenance et prochain raccord

Des images futures effectivement observées peuvent construire une cible après
leur observation. Elles ne peuvent pas entrer dans le déplacement ou la priorité
qui les prédit. Le futur propriétaire devra lier les observations disponibles,
les expériences, les horizons et leurs coûts ; une signature de tenseurs seule
ne certifie pas cette provenance. Une sélection construite avec le futur reste
une sélection de données d'apprentissage après observation, jamais une entrée
disponible au moment de la prédiction ou du choix d'action.

L'essai `ARC4_PRIORITY_MOTION_FIT_PROTOCOL.json` reste P seul. Son résultat,
positif ou négatif, ne raccorde pas m/C et ne ferme pas l'amorçage P/C/ME. La suite
devra proposer un mécanisme de correspondance vérifiable depuis les pixels,
déclarer ses inconnus et ses critères avant l'essai, puis vérifier qu'il fournit
réellement de la supervision utile. Aucun masque, flot ou objet du juge ne sera
transmis à l'agent. Ce texte ne change aucun critère proposé du gate 0.

Sources locales relues : `ragnarok/learning/pixel_prediction_v70.py`,
`counterfactual_targets_v70.py`, `docs/ARC4_COUNTERFACTUAL_TARGETS_DRAFT.md`,
`ARC4_BOOTSTRAP_LIMITS.md`, `ARC4_PREDICTION_OCCLUSION_LIMITS.md`,
`ARC4_FREE_FIELD_SCIENTIFIC_REVIEW_20260910.md`, `ARC4_AGENCY_TIMING_LIMITS.md`
et les revues indépendantes du reconstructeur et de sa tête de priorité.
La distinction des inconnus reprend également la revue contradictoire de
l'agent `audit_motion_bootstrap` du 10 septembre, sans la présenter comme une
validation expérimentale du futur mécanisme.
