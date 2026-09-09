# ARC 4 — assemblage des observations et objectifs propres

`perception_stream_v70.py` assemble le parser, ses instances, le suivi borné et
les objectifs G/I sur un flux de pixels. Les tests utilisent uniquement des
images synthétiques et des gradients CPU. Aucun poids n'est mis à jour, aucun
émulateur n'est ouvert, aucun débit ni résultat d'apprentissage n'est revendiqué.

Le flux reçoit les deux dernières images RGB brutes effectivement observées et
applique leur maximum pixel par pixel. L'appelant fournit une horloge entière en
images ordinaires effectivement émulées, un identifiant de flux et une époque de
mémoire. Chaque environnement ou branche a son propre flux. Les images de
laboratoire et les charges de restauration ne font pas avancer l'horloge de la
partie ordinaire ; une réinitialisation du tracker crée une nouvelle époque.
Les identifiants publics de flux et d'époque sont en lecture seule : un reset
de mémoire décidé par l'agent construit un nouveau flux, avec une mémoire vide.
Chaque ancienne entrée conserve ses identifiants d'origine ; elle ne peut pas
être réétiquetée par les identifiants courants. Un compteur de reset natif/ALE
ne doit pas déclencher automatiquement cette opération ni servir de signal de mort.

Le parser n'a pas d'entrée décrivant la durée. L'assemblage fixe donc d à un
déplacement prédit sur quatre images réelles, conformément au frameskip proposé.
Le suivi convertit ce déplacement en vitesse pour extrapoler selon l'écart réel
des observations. Une transition interrompue après une, deux ou trois images
n'enseigne pas P : elle ne correspond pas à l'horizon prédit. Elle reste une
observation utilisable pour le suivi et G. I exige exactement huit images d'écart,
jamais simplement deux appels qui auraient été tronqués.

La mémoire conserve les pixels uint8 et les observations propres du tracker sur
les huit dernières images de cette horloge. Elle n'accumule pas les graphes de
gradient. Quand une extrémité à huit images existe, son embedding est recalculé
avec les poids courants ; les identités proviennent toujours du suivi observé à
cette date. Cette différence entre poids courants et anciennes décisions de suivi
est explicite. Les copies privées empêchent la mutation d'un résultat retourné
de réécrire la mémoire. Une erreur d'allocation ou d'objectif après affectation
n'avance ni tracker, ni identifiants, ni historique.

Les douze effets décrivent l'action choisie pour l'intervalle suivant ; zéro
signifie inconnu. Le collecteur doit encore garantir que cette action a réellement
été exécutée, que les deux images brutes sont les bonnes, et que les horloges,
flux et époques sont justes. Les arguments de cette API ne prouvent pas cette
provenance. La correspondance des boutons avec les effets reste à apprendre.
C est encore un objectif séparé : lier de façon attestée son effet, son point de
départ et ses deux branches exige l'intégration du collecteur. La gestion de
terminaison, les choix de données, coefficients et mises à jour ne sont pas figés.

La perte P héritée reste inchangée. L'assemblage refuse de rendre une transition
admissible si cette perte n'est pas finie ; des logits individuellement finis
mais extrêmes peuvent faire déborder sa réduction. Le zéro d'une transition
tronquée ne réduit pas de grandes valeurs avant multiplication par zéro. Ce
contrôle de perte ne rembobine pas des observations déjà reçues : un futur
lanceur doit conserver leur comptabilité et arrêter l'apprentissage défaillant.
