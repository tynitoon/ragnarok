# ARC 4 — association apprise sous variation d’exposition

Déclaration prospective du 10 septembre 2026, **avant générateur, données,
modèle et apprentissage**. Un candidat et une tentative ; pas de variante ou
recherche de seuil après résultat. Cette expérience reste synthétique, CPU,
distincte des calibrations Atari closes. Les jeux réservés et le gate 0 restent
fermés. Elle applique les neuf exigences du contrat
`ARC4_SELECTIVE_CORRESPONDENCE_DRAFT.md`, dont le scorer a été vérifié.

## Hypothèse précise

Des marches cycliques entre représentations de pixels peuvent-elles apprendre
une association plus robuste à l’exposition que la même architecture initiale,
tout en s’abstenant sur des points disparus ou indiscernables ? Les positions,
couleurs et expositions d’évaluation sont de nouveaux tirages. Il ne s’agit ni
d’identité physique universelle, ni de segmentation d’objets Atari, ni de ME,
ni d’apprentissage accéléré entre jeux.

L’exposition évite que le succès se réduise à retrouver un RGB identique par
une fonction aléatoire déjà injective. Cette nuisance a été retenue avant données,
sur recommandation de la revue contradictoire. Le facteur d’exposition reste
exclusivement dans le générateur ; il n’est pas fourni au modèle. Une avance
seulement descriptive ou un succès également présent à l’initial ne suffit pas.

## Famille et séparation des données

Chaque clip contient trois images RGB uint8 de **8×8 pixels**. Quatre points
opaques d’un pixel évoluent sur un fond uniforme. Les cinq couleurs de base
sont tirées sans remplacement dans les triples entiers de [32,127]³, avec un
ordre aléatoire. Pour chaque image, un facteur global indépendant uniforme
dans [0,65 ; 1,75) multiplie les couleurs avant arrondi au plus proche.
Le résultat reste dans [21,223] : aucun écrêtage. Les expositions ne sont pas
des paramètres appris, des données du réseau ou des indices de commande.

L’acteur 0 définit la population des requêtes du juge : son pixel visible dans
la première image. Cette identité est invisible au modèle, qui produit une
proposition pour chacun des 64 pixels. La distribution des couleurs et des
positions de cet acteur au départ est la même que pour les autres acteurs.
Les conclusions ne concerneront pas les faux appariements sur tous les pixels
du fond, ni des objets étendus, textures, animations ou transformations de palette.

Au départ, les quatre positions sont tirées sans remplacement parmi 64 cases.
Les positions ultérieures des distracteurs sont tirées sans remplacement parmi
les cases laissées libres par les points dont la position est contrainte.
Chaque strate applique la règle fixe suivante :

- `visible_moving` : acteur 0 visible dans les trois images, à une nouvelle
  position en image 1, puis une position distincte des deux premières en image 2.
  Les trois distracteurs changent librement de place, sans chevauchement final.
- `visible_stationary` : acteur 0 conserve sa position ; les trois autres
  positions sont retirées dans les 63 cases restantes à chaque image.
- `occluded` : acteur 0 est visible en images 0 et 1. En image 2, il occupe
  la même case que l’acteur 1, rendu après lui et opaque. Son identité n’a donc
  plus de pixel visible ; le juge exige une abstention.
- `out_of_frame` : acteur 0 est visible en images 0 et 1, puis sort de l’image
  en image 2. Seuls les trois distracteurs sont alors rendus. Abstention requise.
- `ambiguous` : les acteurs 0 et 1 partagent la même couleur de base et occupent
  les deux mêmes cases durant le clip. Deux tirages Bernoulli indépendants
  décident de leur permutation physique aux images 1 et 2, sans changement RGB
  correspondant. Les deux distracteurs changent de place hors de ces cases.
  Le juge exige l’abstention, même si une proposition tombe par chance juste.

L’apprentissage utilise **128 clips**, la strate visible étant tirée avec
probabilité 1/2 par clip. Aucun masque n’est utilisé pour sélectionner des pixels
dans la perte : le fond et les quatre points y entrent tous. PCG64, graine 19492.
L’évaluation utilise 256 clips i.i.d. par strate, **1 280 au total**, dans l’ordre
du contrat de score ; cinq flux PCG64 issus de SeedSequence(19493). Les familles
ont leurs tirages propres. Les éventuelles coïncidences quantifiées restent dans
les données et seront comptées, sans filtre dépendant du candidat ou du score.

Les RGB d’apprentissage et d’évaluation sont des tableaux séparés. Requêtes,
identités, états physiques, strates et cibles du juge sont conservés séparément.
Le fit ne reçoit que les RGB d’apprentissage. L’inférence ne reçoit que les RGB
d’évaluation, sans juge, strate ou requête. Les propositions sont scellées avant
que le programme du juge ouvre ses cibles et construise les 1 280 verdicts.

## Candidat et objectif

Encodeur partagé par pixel : convolutions 1×1 de canaux **3→32→32→16**, avec
biais, SiLU après les deux premières et normalisation L2 finale, epsilon 1e−6.
Entrée RGB/255 ; aucun indice de position, temps, jeu, bouton ou pixel appris.
**1 712 paramètres** indépendants de la résolution. Initialisation PyTorch
standard, graine 782, float32. Aucun poids préentraîné ou résultat ancien chargé.

Entre deux images adjacentes, les produits scalaires des représentations donnent
une matrice de similarités, puis un softmax par ligne à température **0,07**.
Les marches utilisent A01, A10, A12 et A21. La perte est la moyenne des trois
termes : −log diag(A01 A10), −log diag(A12 A21) et
−log diag(A01 A12 A21 A10), puis la moyenne sur clips et pixels.
Le logarithme borne son argument à 1e−12 pour éviter une valeur non finie.
Aucune cible de correspondance intermédiaire du juge n’entre dans cette perte.

L’objectif adapte le principe de retour au point initial de Jabri et al.,
sans reprendre leur encodeur de patches, leurs données ou leurs performances.
[Principe et limites de la méthode](https://arxiv.org/html/2006.14613v2).
Une permutation fausse peut fermer un cycle. Des pixels d’apparence identique
rendent une partie de la perte irréductible : aucun critère de perte nulle.

**512 mises à jour Adam**, taux 0,001, betas (0,9 ; 0,999), epsilon 1e−8,
aucune pénalité, norme du gradient bornée à 1. Mini-lots de quatre clips dans
l’ordre fixe du tableau : 32 mini-lots répétés durant 16 passages. Pas de dropout,
augmentation supplémentaire, calendrier adaptatif, sélection de checkpoint ou
évaluation intermédiaire. Les deux états initial et final sont conservés.

## Proposition d’association et témoins

Après observation des trois images, P=A01 A12 et Q=A21 A10.
Pour chaque pixel source i, j est l’argmax de P(i,·), avec le premier indice
en cas d’égalité. Proposition admissible seulement si **P(i,j)Q(j,i)≥0,90**
et si la similarité L2-normalisée entre source i et destination j est **≥0,95**.
Sinon, destination `None`. Ces nombres décrivent une règle interne fixe,
pas une probabilité calibrée de visibilité ou d’identité.

Les mêmes propositions sont calculées pour le modèle initial conservé, une fois
que l’entraînement est terminé. Un témoin descriptif externe utilise les RGB/255
directement normalisés L2 comme représentation, avec exactement les mêmes
température et admission. Il n’est ni entraîné, ni une entrée de l’agent. Ses
propositions sont conservées, mais aucun critère n’est choisi à partir d’elles.
Le score primaire reste la comparaison finale/initiale déjà déclarée ; le témoin
explicite ce qui peut être obtenu par normalisation photométrique connue.

Les contrôles d’occultation existants utilisent, dans chaque branche, l’ancre
puis les images aux offsets 2 et 4. Deux familles, raw/pooled, deux branches :
huit clips supplémentaires par état évalué, sans nouvelle émulation. Pour chaque
famille et image_kind, le témoin échoue si l’une des deux branches admet une
association du pixel source (3,2). Les futurs physiques opposés ont déjà été
vérifiés RGB-identiques ; les états du juge ne sont jamais transmis à l’encodeur.

## Budgets et séquence d’exécution

Avant données réelles de ce candidat : revue du présent contrat, implémentation,
fixtures numériques et du générateur sous **60 s CPU cumulées**, avec uniquement
de petits tenseurs/clip de fixtures et un backward de vérification si nécessaire,
zéro mise à jour. Les graines de ces fixtures doivent être distinctes des graines
de production. Revue du code et vérification des sources avant essai.

Une tentative ensuite, **300 s CPU sur un thread**, imports, génération,
512 mises à jour et inférences finales/initiales/témoin comprises ; **64 MB**
d’artefacts. CUDA indisponible, OMP/OPENBLAS/MKL et Torch à un thread. Une réserve
de 15 s est gardée pour la clôture. Le lancement conserve son intention avant
génération. Toute erreur, dépassement ou non-finitude arrête sans recommencer ;
les états partiels, comptes et journaux restent conservés. Pas d’état final
inventé si les 512 mises à jour ne sont pas achevées.

Les sorties comprennent les RGB, métadonnées distinctes du juge, états initial
et final, pertes et comptes de chaque mise à jour, propositions de chaque état
et du témoin, empreintes des sources et des artefacts. Aucune matrice complète
d’attention par scène n’est archivée ; on conserve destinations proposées,
admissions, scores aller-retour et similarités aux destinations retenues.

Après clôture technique et revue, le juge construit une fois les verdicts depuis
les propositions gelées puis applique une fois le scorer prospectif, sous 60 s
CPU distinctes, sans réseau ni nouveau RGB. Les entrées détaillées du score sont
conservées avant ses agrégats. Le contrôle d’intégrité et la revue finale restent
en lecture seule, sans réentraînement. Tous les résultats, positifs ou négatifs,
sont publiés avec leurs dénominateurs et leurs limites.

Un PASS resterait une association sélective dans cette petite famille sous
exposition variable. L’évaluation d’admissions conjointes pour m/C et la question
du conditionnement local avant ME resteraient à traiter. Aucun nouveau raccord
m/C, aucune perception Atari ou transfert ne découle automatiquement de ce fit.
