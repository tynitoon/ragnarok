# ARC 4 — réponse observée et attribution : contrat d’interface

Déclaration prospective du 10 septembre 2026, avant implémentation et fixtures.
Ce jalon d’ingénierie suit la clôture des calibrations locales. Il ne change
aucun protocole expérimental ni seuil antérieur. Les anciennes cibles restent
conservées dans leur périmètre historique.

## Objet et plafond de conclusion

Une différence RGB entre deux branches propres est une observation à des
coordonnées d’écran. Même à la bonne destination prédite, elle n’identifie pas
l’objet responsable. Les deux témoins conservés dans
`tests/test_counterfactual_occlusion_limits_v70.py` donnent des observations
identiques sous des contrôles physiques opposés.

La nouvelle interface sépare donc la réponse observée, la disponibilité de
chaque horizon et l’attribution aux instances propres. Aucun estimateur de
correspondance n’est introduit ici. Toutes les cibles m/C restent inconnues,
avec valeur `None`, poids nul et raison explicite. Ce choix bloque une fausse
supervision ; il n’apprend pas à voir, à contrôler ou à jouer.

## Entrées et frontière temporelle

Une fabrique d’entrée prédictive accepte uniquement les deux images RGB
passées uint8 et l’hypothèse propre à douze effets. Elle en prend une copie
détachée et immuable. Aucun futur, numéro de bouton, jeu, identité vraie,
flot ou masque du juge n’est accepté par cette fabrique.

Une autre fonction, appelée après observation, accepte cette entrée, un seul
`BranchExperience` issu du collecteur propre et les supports des instances
du modèle au départ. Les deux branches appartiennent à ce même enregistrement.
Elle vérifie leur géométrie, le préfixe effectivement disponible, la construction
du max-pooling depuis l’ancre, l’égalité du passé et de l’hypothèse d’effet,
ainsi que la cohérence des identifiants de session, opération et travailleur.
Les supports des instances sont propres, non vides, disjoints et dans l’image.

Les métadonnées relient le résultat à l’opération et à son ancre. Elles ne sont
pas des entrées du réseau. Les numéros de commande restent dans le journal du
collecteur ; l’interface prédictive et les sorties de ce jalon les omettent.
Des égalités de tableaux ou d’identifiants ne prouvent pas l’origine physique
des données. Le propriétaire conserve la responsabilité du paiement, du replay
et de l’ordre temporel. Cette API n’est pas une barrière contre un appelant qui
fabrique un faux enregistrement ou calcule son hypothèse d’effet avec le futur.

## Sortie par horizon

Pour chaque horizon demandé de 1 à 64 frames natives, l’interface indique
séparément si l’image intervention et l’image contrôle sont disponibles.
Si les deux le sont, elle conserve exactement le masque RGB différent à cet
offset. Le choix `raw` ou `pooled` est explicite, sans union entre horizons.
Sinon, la réponse est `None`, jamais une image noire inventée. La longueur
commune et les deux queues non appariées sont conservées.

La cible m concerne le domaine des pixels source, indépendamment de l’existence
d’instances. Elle est inconnue pour correspondance non validée si l’horizon
intervention existe, pour futur intervention manquant sinon. Ce statut ne
signifie pas « fond » ou « immobile ».

Chaque instance conserve ses propres identifiant local et pixels source, avec
une cible C inconnue : correspondance non validée si les deux futurs existent,
horizon commun incomplet sinon. Aucune instance signifie aucune proposition
d’instance, pas un écran déclaré incontrôlable. Aucun zéro supervisé, aucun ME,
aucun poids admissible n’est produit. Aucun constructeur de pertes ou entraînement
n’est ajouté dans ce jalon ; les valeurs `None` empêchent leur usage implicite
comme labels numériques. Les sorties ne partagent aucun tampon modifiable avec
les entrées. Elles sont détachées de tout graphe : uniquement tableaux NumPy et
valeurs immuables, aucune tête ou prédiction future du réseau n’est consommée.

## Vérification déclarée

Fixtures CPU, un thread numérique, **60 secondes cumulées d’exécution des tests**,
imports compris ; arrêt si le budget est épuisé. Zéro entraînement, zéro forward
de réseau, zéro émulation, zéro donnée ou poids expérimentaux ouverts. Aucune
validation conditionnelle ou partie réservée n’est consommée. Les éventuels
échecs de fixtures et corrections restent consignés ; pas de répétition d’une
suite conforme sans modification ou problème nouveau.

Critères instrumentaux, tous requis :

- Conserver exactement une réponse non vide connue et distinguer raw/pooled
  sur un cas où les deux diffèrent. Une implémentation toujours vide doit échouer.
- Respecter les horizons asymétriques, zéro frame disponible et 64 frames,
  sans padding ; m ne dépend pas à tort de la disponibilité du contrôle.
- Les deux paires de mondes physiquement opposés restent indiscernables pour
  l’interface et C reste inconnu, réponse présente ou absente.
- Préserver passé, effet et supports après mutation des buffers de l’appelant ;
  refuser leur modification à travers les sorties. Un futur différent ne
  modifie pas l’entrée prédictive.
- Refuser les mélanges d’ancre/effet, les identifiants incohérents, les traces
  mal formées et les supports d’instances invalides. Aucune donnée du juge
  n’entre dans l’API testée.

Une conformité sur ces fixtures certifie seulement ce contrat. Elle ne valide
pas l’admissibilité d’une seule cible m/C et ne mesure aucune généralisation.
La suite devra proposer et évaluer un mécanisme de correspondance depuis les
pixels, avec sa couverture réelle et ses erreurs, avant un raccord d’apprentissage.

Sources d’autorité : `ARC4_VISION.md` §2.1, 2.2, 2.6 et 6 ;
`docs/ARC4_VISIBILITY_UNKNOWN_DRAFT.md`, `docs/ARC4_BOOTSTRAP_LIMITS.md`.
La revue de conception de audit_motion_bootstrap recommande cette séparation
après les deux contre-exemples d’occultation, sans la présenter comme un amorçage
P/ME résolu. Les choix techniques restent dans l’autorisation actuelle de Jérémie.
