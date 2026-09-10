# ARC 4 — apprendre l’enveloppe des contrastes observés

Hypothèse de développement du 10 septembre 2026, après le survey des actions
et sa revue indépendante. Ce document ne change aucun essai passé, aucun STOP,
ni les critères du gate 0 qui reste non gelé.

## Le problème isolé

Le survey a acquis les 216 comparaisons prévues. Dans 104 paires, au moins un
pixel diffère de la branche NOOP sur les huit images futures. Cinq jeux sur six
présentent de tels écarts aux moments fixés. Cela fournit un signal candidat ;
aucun réseau n’a encore appris à l’utiliser. Les deux fenêtres de Seaquest sont
nulles et doivent rester dans les données. Elles ne disent pas que ce jeu est
incontrôlable à d’autres moments.

Le constructeur de cibles C actuel attend un déplacement prédit et des instances
mobiles. Il peut donc fournir zéro supervision même lorsque les branches
diffèrent. Les cinq fits synthétiques antérieurs, 1 280 mises à jour au total,
ont tous échoué à leur critère global ; leurs résultats restent inchangés.

Nous proposons une tâche auxiliaire distincte de P, C et m : depuis les deux
images passées, prédire les positions d’écran où nos essais ont produit un écart.
Elle ne nécessite ni déplacement prédit, ni tracker, ni identité d’objet.

## Une cible strictement observée

Pour chaque ancre, E(y,x) est le OU logique de
`any_RGB(intervention[a,h,y,x] != NOOP[a,h,y,x])`, pour les dix-huit commandes
constantes a=0…17 et les huit horizons h=1…8 du survey.

La cible reste dans les coordonnées des images futures, sans reprojection sur
un objet supposé. Elle peut inclure un projectile, un objet affecté indirectement,
un score ou du fond révélé. Ce n’est ni un masque physique du joueur, ni m,
ni la contrôlabilité C des instances. Elle décrit ces essais précis, pas toutes
les séquences possibles ni toutes les réalisations des boutons collants.

Le prédicteur reçoit seulement les deux images passées, en RGB natif. Les futurs
servent de cibles, jamais d’entrées. Aucun bouton, identifiant de jeu, coordonnée
absolue, RAM, récompense, objet oracle ou résultat du détecteur classique ne lui
est transmis. L’enveloppe est indépendante de la numérotation des commandes.

La table de codes apprise évoquée dans `ARC4_LOCAL_COMMAND_BOOTSTRAP_DRAFT.md`
serait incompatible avec l’interdiction des embeddings de boutons et des
paramètres dimensionnés par leur nombre dans `ARC4_VISION.md` §2.6. La cacher
à la politique ne suffit pas. Ce candidat de table n’est donc pas implémenté.
L’enveloppe contourne ce besoin pour une tâche auxiliaire seulement ; elle ne
résout ni le choix des actions, ni leur alignement sémantique, ni ME.

## Construction bornée, avant tout fit

Le constructeur sera figé et revu avant sa première lecture des tableaux.
Il parcourra exactement les 216 nouvelles archives du survey, dans leur ordre
déclaré, et conservera les douze ancres entières, y compris les cibles vides.
Aucune autre archive ne sera décodée. Les empreintes des sept tableaux, les
formes, les préfixes complets, les contrôles NOOP répétés, les deux images
passées et tous les comptages RGB bruts seront vérifiés. Une discordance est
un arrêt avec traces préservées, jamais une exclusion silencieuse.

Sortie : douze paires passées et douze masques booléens, avec un manifeste
séparé des entrées du futur réseau. Le manifeste relie chaque cible aux coûts
et aux archives déjà payés ; aucune image n’est réémulée. Une tentative exclusive,
un journal durable, un budget CPU coopératif de 300 secondes après les contrôles
de provenance, et 16 Mo maximum de sorties dérivées sont prévus. Les lectures,
écritures et vérifications des tableaux font partie de ce temps. Aucun modèle,
aucune optimisation et aucun calcul GPU ne sont lancés par ce constructeur.

## Ce que le futur fit pourra établir

Les douze situations ont déjà orienté la conception : elles sont toutes des
données de développement. Les diviser ensuite ne crée pas une validation
indépendante. Un premier fit borné peut seulement vérifier la faisabilité de
l’ajustement et l’apport de l’entrée visuelle contre un réseau identique recevant
une entrée constante. Le réseau initial et la sortie zéro resteront visibles.

Architecture, perte, ordre des exemples, initialisation, seuils, métriques et
budget seront fixés avant tout fit. Les mesures sur cibles non vides seront
séparées des faux positifs sur les vides ; une exactitude dominée par le fond ne
vaudra pas réussite. Les ancres, et non les commandes corrélées d’une même ancre,
constituent les unités. Il n’y aura pas de recherche d’hyperparamètres ni de
prolongation de budget après observation d’un échec.

Une réussite justifierait une validation prospective sur de nouvelles ancres,
choisies avant leurs pixels et dans un budget déclaré. Elle ne démontrerait
encore ni généralisation, ni objets physiques, ni jeu maîtrisé, ni transfert,
ni apprentissage accéléré du jeu suivant.

Références locales : `ARC4_VISION.md` §2.1 et §2.6,
`ARC4_BOOTSTRAP_LIMITS.md`, `ARC4_ACTION_SIGNAL_SURVEY_PROTOCOL.json`,
`craft_v6_out/v70_action_signal_survey/survey_v0.json` (877dc75e…),
et `survey_review_v0.json` (3211472c…).
