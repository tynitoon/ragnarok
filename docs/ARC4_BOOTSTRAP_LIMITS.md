# ARC 4 — ce que le raccord numérique ne résout pas

Point de conception du 9 septembre 2026, avant tout entraînement Atari. Ce texte
ne change ni la vision, ni les critères proposés du gate, ni les protocoles des
essais déjà déclarés.

Le raccord actuel sait conserver les images produites par nos propres commandes,
construire P/C/G/I et vérifier leurs gradients. L'hypothèse d'effet enregistrée
dans les neuf archives natives est encore le vecteur nul de douze valeurs :
inconnu. Les numéros de boutons ne sont pas transmis au parser. Les poids restent
inchangés pendant le diagnostic CPU prospectif ; aucune table de commandes,
reconnaissance de soi ou politique n'en découle.

Il reste une dépendance à résoudre explicitement avant l'entraînement : P/C
utilisent le déplacement prédit sous une hypothèse d'effet ; la découverte des
effets suppose de reconnaître les objets contrôlés ; cette reconnaissance doit
elle-même venir des propres expériences et du parser. Les tests de transport et
de gradients ne démontrent pas que cette boucle peut démarrer.

Deux limites se lisent directement dans l'implémentation :

- Un même écran sous la même hypothèse inconnue fournit le même déplacement au
  parser déterministe, même si les boutons enregistrés diffèrent. Il ne peut donc
  pas représenter deux conséquences différentes uniquement grâce à ces entrées.
  Cela n'interdit pas un apprentissage moyen ; cela ne valide pas une prédiction
  conditionnée par la commande avant la découverte de l'adaptateur.
- Les cibles C restent inconnues lorsque le mouvement propre prédit est absent,
  la paire trop courte ou sa projection hors écran. Des branches visuellement
  différentes ne garantissent donc pas des cibles C exploitables. De même, les
  objectifs de séparation G et d'identité I peuvent avoir un gradient nul lorsque
  les représentations se confondent. Le garde de norme protège un calcul fini ;
  il ne crée ni supervision ni information manquante.

La prochaine décision d'architecture devra préciser l'amorçage de l'adaptateur
et l'ordre des mises à jour, puis figer ces choix avant la collecte d'entraînement.
Les paires de huit images déjà prévues peuvent servir à examiner les délais de
divergence, mais C au quatrième écran ne suffit pas à établir le contrôle au
premier écran décrit pour ME. Les boutons collants et les préfixes tronqués
doivent garder un état inconnu, sans devenir des preuves d'absence de contrôle.
Les éventuels tests synthétiques devront garder leurs étiquettes dans le juge,
jamais dans les entrées du parser ou de l'adaptateur.

Sources locales : `ARC4_VISION.md` §2.1–2.2 et §11 ;
`ragnarok/learning/parser_v70.py`, `counterfactual_targets_v70.py`,
`grouping_v70.py`, `identity_v70.py`,
`archived_pixel_diagnostics_v70.py` ; revues synthétiques G/I et de la session
temporelle. Les dépendances de démarrage ci-dessus sont des limites à tester,
pas un résultat négatif d'apprentissage mesuré.
