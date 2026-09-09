# Admission des gradients avant Adam — proposition non gelée

9 septembre 2026. La revue du raccord temporel a rencontré, sur un parser réel
CPU et des images synthétiques avec un départ noir, une perte I de 3,4748144 mais
un gradient maximal de l'ordre de 2,27e22 (norme L2 de 6,0126643e22, seed Torch764).
Les poids n'avaient pas été mis à jour. La recette est conservée dans
`v70_ordinary_rgb_trace/temporal_session_review_v0.json`, SHA f7d44de2….
Un gradient
fini ne suffit donc pas : son carré peut déborder les moments float32 d'Adam.

`gradient_guard_v70.py` propose une borne globale de norme **1**, avant toute
mise à jour. C'est un choix prospectif d'optimisation à revoir et inscrire dans
le protocole avant un entraînement Atari ; aucun seuil de résultat ne change.
La norme utilise des carrés et réductions float64. La multiplication par le
facteur de réduction se fait aussi en float64 avant le retour en float32, pour
ne pas annuler prématurément un très petit facteur. Le reçu rend les normes
avant/après, le facteur, le nombre de tenseurs/éléments et le fait de réduire.

Le module vérifie les poids et tous les gradients denses float32, puis prépare
tous les gradients candidats et le reçu avant de les affecter. Une erreur de
validation ou d'allocation durant cette préparation conserve les anciens
gradients. Le propriétaire doit exclure les accès concurrents et **ne pas
exécuter Adam après un refus**. Il n'y a ni backward, ni step, ni changement des
poids/moments dans ce helper. Il ne constitue pas un optimiseur transactionnel
et ne récupère pas une panne matérielle pendant l'affectation des gradients.

La borne <=1 réduit le risque de débordement lors de leur mise au carré. Elle ne
garantit pas la stabilité de poids ou de moments déjà corrompus, une convergence,
un gradient utile, ni une efficacité d'apprentissage. Un gradient nul peut être
admis ; une borne positive inférieure à ce que float32 peut représenter peut
même annuler tous les gradients après conversion. Le défaut proposé ici reste
1 ; aucune utilité d'une telle borne extrême n'est revendiquée. Le futur
ordonnanceur doit distinguer l'existence d'un objectif valide
du simple fait d'obtenir des nombres finis. Les paramètres Adam envisagés par
la vision restent lr3e-4 ; coefficients des objectifs, microbatching, calendrier,
fréquence et effets du clipping doivent être explicités avant le gel. La fraction
de mises à jour réduites et les normes initiales devront être rapportées.

Vérification locale sur CPU uniquement, sans données Atari ou de calibration.
Un test reproduit ce gradient du vrai parser puis effectue une seule mise à jour
Adam sur le témoin synthétique après admission : poids et moments restent finis.
Ce contrôle mécanique ne montre ni convergence ni apprentissage d'un jeu.
Le code ne fournit aucun lanceur d'entraînement et n'autorise aucun gate.
