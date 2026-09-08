# Ragnarok — point de reprise Codex

Mis à jour le 8 septembre 2026, lors du passage de la session Claude importée à
une conversation Codex légère. Ce document est un index de reprise, pas un
nouveau protocole expérimental.

## Objectif et manière de travailler

Jérémie veut construire son propre agent, sobre en ressources, qui apprend de
ses expériences, garde ses acquis et les réutilise pour apprendre plus vite.
La capacité à parler est secondaire. La démonstration doit être compréhensible
et observable ; battre une heuristique ou maîtriser tous les objectifs n'est
pas le critère central.

La vision récente, ARC 4, est de jouer à partir de l'écran et de commandes
simulées, en progressant Atari → NES → Mega Drive/SNES → PC. Mesurer le bénéfice
de l'expérience passée sur des jeux nouveaux, puis son accumulation. Cette
vision est une ambition ; la compétence générale et les étapes PC/3D ne sont
pas démontrées. Jérémie confie les choix techniques à l'agent et attend qu'il
teste ce qu'il construit. Les résultats négatifs et les limites restent publics
dans le dossier scientifique.

## État de la reprise

La dernière instruction dans Claude demandait une pause pour préserver les
tokens destinés au projet Tavern. L'audit ARC 4 et la surveillance du gate v62
avaient été arrêtés. Le calcul GPU du gate pouvait terminer seul ; aucun
verdict ne devait être écrit ou annoncé avant une demande de reprise.

Dans cette conversation, Jérémie a explicitement confirmé « Reprendre le travail
sur Ragnarok » le 8 septembre au soir. La pause est levée : examiner le gate v62,
consigner son résultat et poursuivre selon les règles gelées ; reprendre aussi
la revue de conception ARC 4. La récupération du contexte est faite.

Vérifications locales au moment du transfert :

- Branche `master`, dernier commit `90b304a` : brouillon `ARC4_VISION.md`.
- Les résultats `craft_v6_out/v61_gate_v62.json` et
  `craft_v6_out/v61_gate_v62_run.log` existent, dernière écriture le 8 septembre
  à 06:00 heure locale. Après levée de pause, `PROCEED` a été recalculé et confirmé
  indépendamment : huit checkpoints identiques au JSON, courbes complètes,
  REACHABLE/FRESH-LEARNS/ROOM et CONSISTENT passent. Résultat et budget consignés
  dans `ARC3_PLAN.md` §13 et en fin de `preregistration.md`.
- Des JSON/checkpoints de smoke tests et du gate, ainsi que `.codex/`, étaient
  déjà non suivis. Ne pas les inclure dans un commit documentaire.
- Aucun fichier `scripts/*v70*` trouvé : les modules décrits dans ARC 4 restent
  une liste d'implémentation prévue, pas une implémentation validée.

## Résultats antérieurs à préserver

- L'arc de nécessité v49–v57 est clos après les résultats négatifs gelés. Les
  anciennes affirmations fortes de compounding ont été corrigées après découverte
  d'une fuite du graphe des recettes dans les observations.
- ARC 2 est clos sur un résultat partiellement négatif : transfert sans mise à
  jour des poids, 3/6 objectifs contre 0/6 ; apprentissage accéléré non détectable,
  delta 0,069 contre bruit 0,076. Source : `ARC2_PLAN.md` §15.
- ARC 3 v61 a amélioré le substrat avec une loi chimique cachée partagée entre
  mondes. Ses deux gates ont échoué en faisabilité, après environ 6,7 GPU-h.
  Aucun verdict sur le bénéfice d'un agent expérimenté n'en découle.
- Jérémie a ensuite autorisé le redesign v62. Le README précède cette reprise :
  pour l'état v62, utiliser `ARC3_PLAN.md` §12, surtout §12.6.

## ARC 3 v62 : reprendre sans changer les règles

Sources : `ARC3_PLAN.md` §12–12.6, fin de `preregistration.md`, commit de gel
`a189ea5` et tag `v62-stage0-freeze` (tag à vérifier avant utilisation).

Le store persiste entre objectifs d'un même monde pour toutes les conditions.
Ceci remplace la règle initiale v61 de store vide par objectif. Mondes de gate :
8102/8103 ; trois conditions fresh ; six rounds ; 64 environnements.

FRESH-LEARNS exige que les trois conditions atteignent 0,6 au niveau 2 en deux
rounds, et qu'au moins deux sur trois atteignent 0,6 au niveau 3 en six rounds,
sur chacun des deux mondes. Une défaillance de FRESH-LEARNS clôt la ligne ARC 3 :
pas d'assouplissement supplémentaire. Le plan conserve une seule modification
durcissant le problème si ROOM échoue ; appliquer exactement sa règle, sans
inventer une nouvelle relance. Le contrôle CONSISTENT peut arrêter K1.

Le plafond total ARC 3 est de 42 GPU-h. §12.6 corrige l'ordre des coupes :
réduire d'abord G' à une unité, puis B_test de 6 à 5. Le budget utilise les temps
réellement mesurés du gate. La confirmatoire est N=9, trois lignées ; ne pas
couper une lignée, Fb ou le probe.

Décision vérifiée à la reprise : B_test=6, N=9, G' sur deux unités. Projection
totale 31,9926 GPU-h ; gate réel 4,4789 h, préentraînement estimé 4,4281 h, probe
prévu 0,5 h. Les 11 tests ARC 3 passent (137,15 s, comptés en plus par prudence).
Un lanceur séparé `scripts/resume_v62.py` prépare les trois préentraînements puis
le probe, avec arrêt après celui-ci pour revue. Son état sera dans
`craft_v6_out/v62_resume_state.json` et ses logs dans
`craft_v6_out/v62_resume_stdout.log` / `v62_resume_stderr.log`.
Vérifier cet état et les processus avant toute nouvelle exécution ; ne jamais
démarrer un deuxième entraînement en doublon.

Après le gate validé, suivre les commandes gelées §12.6 :
préentraînement des trois lignées, probe, puis confirmatoire seulement si le
probe autorise la suite, score et figure. Si STOP, consigner honnêtement et ne
pas lancer la suite. Les valeurs par défaut de `gate_v61` visent les anciens
mondes et peuvent écraser leurs fichiers : ne pas l'exécuter sans les arguments
gelés. « Publier » ici ne vaut pas autorisation d'envoyer ou pousser à distance.

## ARC 4 : prochain travail de conception

`ARC4_VISION.md` est un brouillon, non gelé, issu d'un panel de conception et
d'un premier audit ; réponses dans sa §10. Le deuxième audit avait été interrompu
sous Claude ; une nouvelle passe indépendante est réalisée lors de cette reprise.
La note Claude référence `workflows/scripts/arc4-vision-audit-wf_9393b889-a8b.js` :
emplacement réel et compatibilité non vérifiés, ne pas supposer ce workflow
disponible dans Codex.

Après levée de pause : reprendre la revue du brouillon avant implémentation et
gel du gate 0. §6 décrit le travail prévu. Le premier gate examine les primitives
sur une référence externe, puis la perception apprise à partir d'expériences
propres. Plafond proposé pour gate 0 : 12 GPU-h ; le reste des budgets et des
estimations doit être vérifié avant engagement.

Les pixels et les boutons sont les seules interfaces agent ; RAM, objets et
scores vrais restent côté évaluateur. Aucune procédure manuelle ne doit fournir
gratuitement la compétence coûteuse dont on prétend mesurer le transfert.

## Sources de récupération

- Dépôt : `README.md`, `ARC2_PLAN.md`, `ARC3_PLAN.md`, `ARC4_VISION.md`,
  `preregistration.md`, `RESEARCH_DIRECTION.md`, `scripts/INDEX.md`, `SETUP.md`.
- Mémoire Claude locale :
  `C:/Users/jeremie/.claude/projects/F--dev-ragnarok/memory/MEMORY.md`.
  Lire à la demande `goal_north_star.md`, `goal_ladder_arc4.md`,
  `feedback_ownership_arc4.md`, `project_arc3_alchemy.md` et les retours utilisateur.
  Ces notes contiennent aussi des états anciens : les amendements gelés récents
  priment pour les détails expérimentaux.
- Session importée consultable par pages :
  `01a082c3-538e-75a2-b196-4774330a51b8`.
- Conversation de reprise : `01a082c5-c85d-7031-95e4-8d432972cb6e`.

Ne pas recharger les centaines d'échanges importés. Mettre à jour ce point de
reprise après chaque transition importante, avec les résultats, décisions et
prochaines actions vérifiés.
