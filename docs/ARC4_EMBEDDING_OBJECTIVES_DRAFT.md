# ARC 4 — groupement et identité issus des propres prédictions

Prototypes CPU synthétiques, non gelés pour une expérience. Ils rendent les
objectifs G et I du plan explicites ; ils n'établissent aucune perception apprise.
Les coefficients des pertes, les données et le calendrier d'entraînement restent
à arrêter avant le gate. Aucun enseignant RAM, référence classique ou archive
de calibration n'intervient dans ces modules ou leurs tests.

## G : destin commun local

La cible part des logits de premier plan du parser, strictement positifs. Deux
pixels voisins horizontaux ou verticaux appartiennent au même groupe si leurs
déplacements prédits diffèrent d'au plus 0,5 pixel, pour l'horizon explicitement
déclaré. Cette tolérance concerne le déplacement sur l'intervalle, pas une vitesse
par image. Le protocole devra fixer l'horizon et cette valeur avant utilisation.
Les composantes du graphe définissent les groupes, sans utiliser f pour fabriquer
sa propre cible. Une chaîne de petits écarts peut relier deux extrémités dont les
mouvements diffèrent davantage ; ce n'est pas une borne sur le diamètre du groupe.

Les cibles NumPy sont détachées. La perte ne fait donc pas apprendre m ou d par
cet objectif. Elle retourne deux termes séparés : moyenne par groupe de la
variance euclidienne de f ; moyenne entre paires de groupes de
max(0, 1 − distance entre leurs moyennes)². Le poids par groupe est égal, sans
privilégier les gros objets par leur nombre de pixels. Les pixels hors cible
ont un gradient nul. La limite proposée de 512 groupes provoque un refus explicite,
jamais une sélection silencieuse ; aucune complexité mémoire réelle n'est mesurée.

Limites : m et d peuvent être faux, des objets peuvent fusionner ou se fragmenter,
des objets séparés mais similaires sont poussés à se distinguer. Un effondrement
exact où toutes les représentations sont identiques est un point stationnaire
de cette marge euclidienne ; le test conserve ce contre-exemple. Les poids
aléatoires initiaux ne garantissent pas qu'un tel effondrement sera évité.

## I : correspondance visible à huit images

Deux sorties du propre tracker, séparées d'exactement huit unités de son horloge,
fournissent les correspondances. Le collecteur doit définir cette horloge en
images effectivement émulées, tenir compte des transitions tronquées et fournir
deux images alignées avec leurs sorties. Chaque environnement ordinaire ou
branche de laboratoire a un identifiant de flux distinct. L'époque change quand
la mémoire du tracker est réinitialisée, même si ses identifiants repartent à zéro.
Ces métadonnées sont internes au collecteur, jamais des observations oracle.

La perte utilise les embeddings recalculés avec gradient sur les pixels réellement
observés aux deux extrémités. La moyenne de chaque instance est normalisée, puis
InfoNCE compare un identifiant partagé aux autres identifiants visibles à l'autre
extrémité, avec température proposée de 0,1. Les deux directions sont moyennées
quand chacune dispose d'au moins deux candidats. Une direction sans négatif ne
compte aucune ancre supervisée. Les objets absents seulement maintenus par le
tracker n'apportent ni positif ni négatif ; un identifiant neuf n'est jamais forcé
à correspondre. Les correspondances numériques sont refusées entre flux/époques.

Limites : le tracker est lui-même guidé par f, donc les cibles ne sont pas une
vérité indépendante. Un échange d'identité ou une fragmentation peut enseigner
une erreur. Un objet revu après une absence reste admissible si le propre tracker
lui rend son identifiant. Les tokens de flux/époque ne prouvent pas à eux seuls
la provenance des images : leur attribution et l'alignement restent à intégrer
et vérifier dans le collecteur. Aucun décodeur de jeu ne corrige ces correspondances.
