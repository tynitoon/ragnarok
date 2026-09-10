# Diagnostic des contradictions locales du premier fit Atari

10 septembre 2026. Diagnostic choisi **après** le résultat négatif du fit,
avant tout calcul de ces équivalences sur ses images. Ce n’est ni une nouvelle
validation, ni un nouvel essai d’apprentissage. Les critères du fit restent
inchangés et sa validation conditionnelle n’est pas déclenchée.

## Question et périmètre

Le FCN à cinq convolutions dilatées 1/2/4/8/16 ne reçoit que deux RGB passés.
Des entrées locales indiscernables peuvent-elles porter des étiquettes futures
opposées ? Quantifier ces contradictions donne des bornes de performance,
sans supposer que les 10 177 paramètres peuvent atteindre ces bornes.

Lire uniquement le dérivé déjà revu `v70_response_envelopes/targets_v0.npz` :
12 paires passées et 12 masques, toutes les situations de développement,
dans l’ordre conservé. Aucun accès aux 216 archives sources, à un émulateur,
aux poids ou à de nouveaux jeux. Aucune inférence, mise à jour ou recherche
de seuil. Les empreintes des sources, résultats et revues précédents restent
vérifiées avant cette lecture. Le juge externe ne devient jamais un composant
du réseau ou de sa supervision.

## Équivalence suffisante, y compris aux bords

Le rayon réceptif total est 31. Pour chaque pixel, la clé comprend le patch
63×63 des six canaux ordonnés, répliqué aux bords de l’image brute, et les
quatre distances haut/bas/gauche/droite, chacune écrêtée à 31. Ces distances
sont des informations du juge seulement. SHA256 indexe les candidats ; une
comparaison de **tous les octets** résout chaque collision avant regroupement.
Un groupe conserve les coordonnées d’un représentant et les IDs de ses pixels,
pas une copie de tous les patches. Aucun rapprochement approximatif.

Un padding global seul ne remplace pas les paddings de chaque couche. La
topologie jointe à la clé règle ce problème : en remontant les couches, les
déplacements maximaux sont 16, 24, 28, 30, 31. Tout clamp est donc évalué dans
ce rayon ; écrêter une distance à 31 ne modifie aucun clamp accessible. Les
mêmes feuilles et les mêmes clamps donnent les mêmes activations par induction.
Toucher le bord à distance 31 est possible, le franchir ne l’est pas. Cette
preuve vaut pour l’opérateur mathématique local déclaré, pas pour une garantie
d’identité bit à bit entre différents backends flottants. Scinder davantage
des entrées équivalentes reste une relaxation sûre des contraintes.

Les témoins synthétiques vérifient les clamps 1D, une évaluation récursive
indépendante, l’ordre temporel, la résolution des collisions, et un contre-
exemple au padding global seul. Aucun réseau Torch n’est construit.

## Deux bornes distinctes

Pour l’ancre non vide a, P_a est le nombre total de positifs ; p_ag et n_ag
sont les effectifs positifs et négatifs du groupe g. Tout masque constant
dans chaque groupe a une IoU `sum(p_ag*x_g)/(P_a+sum(n_ag*x_g))`. Le maximum
est obtenu parmi les préfixes triés par p/n, les groupes positifs sans négatifs
en premier. Les comparaisons et le maximum sont rationnels exacts. On autorise
des choix **indépendants entre ancres**, puis on reprend les moyennes par ancre
et par jeu du score gelé. C’est une borne supérieure relâchée, pas un masque
réalisable par le réseau. Les ancres vides ont une IoU indisponible (`null`).
La fraction rationnelle du macro-score permet de comparer exactement à 1/4.
Une borne >=1/4 ne démontre aucune faisabilité et ne disculpe pas les ambiguïtés.
La contrainte de faux positifs des ancres vides n’est pas optimisée conjointement.

Pour la BCE, le regroupement est global entre ancres. Avec A ancres et N
pixels, un pixel positif d’une ancre mixte porte le poids 1/(2*A*P_a), un
négatif 1/(2*A*(N-P_a)). Une ancre à une seule classe donne 1/(A*N) par pixel.
Les masses u_g et v_g déterminent l’infimum de contribution
`-u_g*log(u_g/(u_g+v_g))-v_g*log(v_g/(u_g+v_g))`, somme sur tous les groupes.
Les classes pures contribuent zéro à l’infimum, éventuellement atteint seulement
avec des logits infinis. Le calcul utilise float64 et `math.fsum`. Il ne
garantit pas l’optimum commun aux critères IoU et faux positifs.

Un écart à ces bornes ne prouve ni minimum local, ni manque de capacité, ni
nécessité d’allonger l’entraînement. Des labels contradictoires pour ce voisinage
ne prouvent pas que la perception ou le contrôle d’un jeu sont impossibles.

## Exécution unique et bornée

CPU un seul thread, aucun GPU. Maximum 300 s de travail, avec réserve de clôture
10 s incluse. Le travail commence après fsync de la tentative exclusive et
inclut décodage du dérivé, groupes, métriques, NPZ, journal fermé, payload fsync
et hash, ainsi qu’une nouvelle vérification finale des sources, preuves et du
préflight contre leurs empreintes initiales. L’empreinte initiale du préflight
reste celle attestée dans la tentative et le résultat. La fermeture du journal
est tentée même si l’écriture de son dernier événement échoue.
Imports et preuves avant tentative, petit reçu final et console après
mesure sont exclus. Contrôles coopératifs chaque rangée et entre phases ; un
dépassement final donne `incomplete`, même si les tableaux sont complets.
Pas de retry ni d’extension après une tentative. Aucun résultat partiel ne
devient une borne sur l’ensemble des situations.
`bounds_usable` est vrai seulement pour une exécution finale complète ; le
statut candidat d’un payload antérieur ne suffit pas.

Au plus 16 MB de sortie binaire, plus reçus JSON et journal borné. Les tableaux
stockés sont les IDs des groupes et leurs représentants ; les données originales
restent inchangées. Avant le lancement : sources et protocole committés,
tests synthétiques et revue indépendante, reçu de préflight committé.
Après : préserver les sorties, revue indépendante des groupes et bornes,
en distinguant exécution valide et diagnostic scientifique non concluant.

Commande fixe : `C:/Python314/python.exe -m scripts.diagnose_response_envelope_equivalence_v70 --run`
avec CUDA_VISIBLE_DEVICES vide et OMP/OPENBLAS/MKL_NUM_THREADS=1.
