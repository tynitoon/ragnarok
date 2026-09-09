# Diagnostic des objectifs sur les propres traces archivées

Préparation du 9 septembre 2026. Aucun appel au modèle sur les archives natives
n'a encore été réalisé. `ArchivedPixelDiagnostics` calcule les objectifs P/C/G/I
et leurs gradients sur CPU, avec le parser détenu exclusivement par le diagnostic.
Il ne construit aucun émulateur ni optimiseur et ne modifie aucun poids.

Le futur lanceur devra vérifier les empreintes des traces et leur revue native
avant de les charger, puis figer les cas, positions, initialisation et critères
numériques avant la première inférence. Le module seul n'atteste pas la provenance
des pixels qui lui sont fournis. Aucun résultat de perception ou d'apprentissage
ne découle d'un gradient fini, non nul ou limité à une norme de un.

Chaque objectif P part des deux vrais écrans de départ de la commande enregistrée.
Un préfixe inférieur à quatre images reste inconnu pour P. La mémoire G/I reçoit
seulement les observations effectivement archivées : un trou de collecte ouvre
une mémoire neuve à son vrai écran de départ, sans inventer les images manquantes.
Au sein d'un bloc contigu, seuls les écrans courants déterminent une rupture ;
une mort native ne fournit jamais d'étiquette. Une source réutilisée pour amorcer
la mémoire n'est pas comptée comme une nouvelle image payée ni une cible en plus.

Le déplacement P/C reçoit l'hypothèse d'effet de douze valeurs déjà enregistrée.
Les observations temporelles restent sous hypothèse inconnue. Les numéros de
boutons sont des métadonnées propres et ne sont jamais transmis au réseau.
C est calculé au quatrième écran poolé de la paire payée ; une paire tronquée,
sans mouvement propre ou sans pixels supervisables peut ne fournir aucune cible.

La somme à coefficients unitaires sert uniquement à exercer ensemble les graphes
P, foreground, G et I d'un pas, puis C séparément. Ce n'est pas une recette
d'entraînement figée. La norme est mesurée en float64, le garde de gradient déjà
relu est appliqué, les gradients sont effacés, et l'empreinte des poids doit rester
identique. Toute erreur bloque la suite du diagnostic. Une revue et un protocole
prospectif du lanceur demeurent nécessaires avant les données natives réelles.
