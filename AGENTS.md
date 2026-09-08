# Ragnarok — consignes de reprise

Lire d'abord `docs/CODEX_HANDOFF.md`, puis uniquement les sections des plans utiles
à la tâche. Le document indique l'état de pause et les sources à consulter. Les
instructions actuelles de Jérémie priment sur cet état historique.

## Cap du projet

Démontrer qu'un agent apprend et réutilise ses acquis pour apprendre plus vite.
La vision ARC 4 est un agent qui joue depuis les pixels et les boutons, avec une
progression Atari → NES → 16 bits → PC. Les ambitions et les résultats établis
doivent toujours être distingués. Une avance initiale ne prouve pas une vitesse
d'apprentissage supérieure.

## Travail et preuves

- Prendre en charge les choix techniques, l'implémentation et les vérifications
  pertinentes ; communiquer en français simple.
- Respecter les protocoles gelés, les budgets et les règles STOP. Pour ARC 3 v62,
  `ARC3_PLAN.md` §12.6 corrige les sections antérieures et certaines notes Claude.
- Ne pas modifier après observation les critères d'un résultat. Garder les
  résultats négatifs, distinguer faisabilité, transfert et apprentissage accéléré.
- Aucune information oracle dans les observations de l'agent. Pour ARC 4, la RAM
  de l'émulateur est réservée à l'évaluation externe.
- La préférence historique est une revue contradictoire par plusieurs agents
  aux jalons scientifiques majeurs ; ce n'est pas nécessaire pour une simple
  mise à jour documentaire.
- Préserver les fichiers, résultats et changements existants. Ne rien supprimer
  sans accord explicite. Faire des commits ciblés aux étapes cohérentes ; ne pas
  inclure les configurations locales ou les sorties d'autres travaux.

## Contexte et environnement

Ne pas charger l'intégralité de la session Claude ni tous les logs. Maintenir le
point de reprise court ; lire les sources par sections. Ne pas afficher le
contenu des configurations locales : elles peuvent contenir des identifiants.

`SETUP.md` décrit les environnements Python ; vérifier l'interpréteur et les
dépendances avant un lancement. Les commandes gelées ARC 3 sont dans le plan,
pas dans les valeurs par défaut des scripts. Les anciens arcs restent archivés.
