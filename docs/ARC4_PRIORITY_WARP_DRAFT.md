# Candidate de reconstruction : priorités lors des recouvrements

Opérateur distinct, sans réseau ni entraînement. Les anciens reconstructeurs,
objectifs, protocoles et résultats restent inchangés. Ce prototype répond au
défaut du rectangle opaque documenté dans ARC4_PREDICTION_OCCLUSION_LIMITS.md.

L'équation 13 du [papier de Niklaus et Liu, CVPR 2020](https://arxiv.org/html/2003.05534v1)
pondère les contributions projetées par l'exponentielle d'une importance avant
normalisation. La priorité relative peut ainsi départager les pixels qui se
recouvrent. Leur système interpole entre deux images connues, utilise un flot
pré-entraîné et estime l'importance à partir de leur accord photométrique.
Au §3.1, ils signalent aussi un échec de convergence de l'importance prédite
depuis une seule image. Ce résultat ne valide donc pas notre apprentissage
prospectif depuis les seules observations passées. Nous reprenons uniquement
le principe de l'opérateur, dans une implémentation PyTorch indépendante ;
aucun code CUDA, poids ou estimateur photométrique de leur système n'est utilisé.

Contrat de la candidate : RGB source, déplacement propre et log-priorité propre,
tous en float32, finis, sur un même appareil ; la priorité a une composante par
pixel et reste explicitement dans [-20,20], sans correction silencieuse. Plus
elle est grande, plus le pixel contribue à la couleur lorsqu'il rencontre un
autre pixel. Une priorité nulle partout retrouve la formule uniforme.
Les trous restent noirs : leur contenu nouveau n'est pas résolu par cet opérateur.

Le mélange normalisé des couleurs est atténué uniquement lorsque la couverture
géométrique tombe sous 1e-6. Le calcul utilise des sommes pondérées en float64
pour conserver les très petits poids géométriques même avec une priorité -20.
Les RGB et la couverture géométrique reviennent en float32 ; la couverture
pondérée reste float64 et l'indicateur de plancher est booléen. Une log-priorité
constante ne doit pas assombrir le mélange. La tolérance des témoins z=0 est
2e-7 sur RGB, 1e-6 sur couverture pondérée ; ce n'est ni une identité bit à bit
avec l'ancienne accumulation, ni une borne universelle pour toute image. Aucune
mesure ancienne n'est réattribuée à cette variante. Ces couvertures ne sont ni
des probabilités de visibilité, ni une détection de premier plan.

La revue de la première version, commit 7d14f93, a trouvé un défaut avant tout
entraînement : son plancher était relatif au maximum des priorités contributrices.
Sur deux pixels blancs de priorités -20 et +20, le second juste hors écran ne
contribue pas. Le faire entrer de 2^-23 pixel faisait passer la couleur du premier
de 1 à 0,11920929, alors que sa couverture était déjà au moins 1. Le maximum
changeait brutalement et provoquait une atténuation injustifiée. Le test de
régression a reproduit l'échec, puis la normalisation a été corrigée en séparant
mélange et couverture géométrique. L'usage interne du float64 et les tolérances
ci-dessus font partie de cette correction préalable, sans nouveau fit ni critère.

Les tests imposent au juge un rectangle opaque devant un fond immobile et des
priorités appropriées : ils vérifient seulement que l'opérateur peut représenter
ce cas sans déplacer le fond pour éviter le gris. Ces priorités déclarées ne
sont jamais des entrées de modèle, des cibles d'entraînement ou des priorités
apprises. D'autres tests vérifient les gradients sans aucune mise à jour,
les recouvrements inversés, les trous, les bornes et les décalages communs.

Avant tout usage dans un agent, les priorités devront être prédites uniquement
depuis ses propres observations disponibles et l'effet choisi. Le futur pourra
superviser une perte mais pas fournir la priorité utilisée pour le reconstruire.
La provenance ne peut pas être attestée par une simple signature de tenseurs :
le raccord propriétaire devra la garantir. Aucune tête de priorité, collecte,
perte complète ou recette d'apprentissage n'est encore intégrée à cette candidate.

La correction de capacité du reconstructeur ne résout pas l'identification du
mouvement ni l'amorçage. Une priorité erronée peut encore masquer un mauvais
déplacement. Le précédent `motion_foreground_target` ne doit pas être raccordé
comme si l'ambiguïté des correspondances occultées était résolue : la définition
des cas inconnus pour m/C reste à concevoir et à vérifier séparément. Aucun
critère de gate Atari ni d'ajustement antérieur n'est modifié par ce prototype.
