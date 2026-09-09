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
- À la récupération initiale, aucun fichier `scripts/*v70*` n'existait. Depuis,
  le premier module CPU a été implémenté et vérifié ; état actuel ci-dessous.

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
Le lanceur séparé `scripts/resume_v62.py` exécute les trois préentraînements puis
le probe, avec arrêt après celui-ci pour revue. Ses tests CPU couvrent les sorties
incomplètes, les erreurs, les plafonds et les arrêts scientifiques. Son état est dans
`craft_v6_out/v62_resume_state.json` et ses logs dans
`craft_v6_out/v62_resume_stdout.log` / `v62_resume_stderr.log`.
Vérifier cet état et les processus avant toute nouvelle exécution ; ne jamais
démarrer un deuxième entraînement en doublon.

Lancement confirmé le 8 septembre 2026 à 23:02:59, heure de Paris : contrôleur
PID 29812, premier enfant prétrain s0 PID 28304, état `running/pretrain_s0`.
Ces PID sont un relevé de démarrage, pas une preuve que les processus tournent
encore : vérifier l'état et les lignes de commande au prochain tour. Les trois
préentraînements puis le probe sont enchaînés automatiquement par le lanceur,
sans autre appel de modèle. Il s'arrête après le probe pour revue, même si le
probe passe. Un suivi horaire est maintenant actif dans cette tâche : automation
`suivi-ragnarok-arc-3`, « Suivi Ragnarok ARC 3 ». Il reste silencieux pendant la
progression normale et reprend la revue à la fin du probe. Aucun lancement
confirmatoire n'a encore eu lieu. Ce suivi doit se mettre en pause à la clôture
de l'ARC 3 ou si une décision de Jérémie devient nécessaire.
Les 16 tests CPU du lanceur passent, en plus des 11 tests ARC 3.

Dernier relevé du 9 septembre à02:43 : lineage0 terminée,4mondes/M1–M4 vérifiés,
8090s dans sonlog (8100,22s contrôleur). Contrôleur29812 actif, nouveauprétrain
lineage1 PID27396 en cours, deux mondes terminés (M2 à4419s), troisième entamé.
Projection actualisée≈34,35h sous42h, B_test6/N9/G'2
conservés ; pasdeprobe ni deconfirmatoire. Reçu `v62_pretrain_s0_receipt.json`.
Lire les logs réels `v61_pre_s0.log` puis s1/s2 :
les sorties des enfants ne remontent pas dans le log stdout du contrôleur Windows.

Après le gate validé, suivre les commandes gelées §12.6 :
préentraînement des trois lignées, probe, puis confirmatoire seulement si le
probe autorise la suite, score et figure. Si STOP, consigner honnêtement et ne
pas lancer la suite. Les valeurs par défaut de `gate_v61` visent les anciens
mondes et peuvent écraser leurs fichiers : ne pas l'exécuter sans les arguments
gelés. « Publier » ici ne vaut pas autorisation d'envoyer ou pousser à distance.

## ARC 4 : instruments CPU, aucun gate ni apprentissage lancé

`ARC4_VISION.md` reste un brouillon, non gelé. Jérémie autorise sa préparation
pendant ARC 3 et demande de poursuivre d'un jalon au suivant sans relance.
Revue de conception : `reviews/2026-09-08-arc4-second-audit.md`.

Environnement `.venv-arc4`, Python 3.14 / ALE 0.12.1 / NumPy 2.4.4 / SciPy 1.18.1,
sans Torch ; ne pas modifier l'environnement GPU ARC 3. CPU réel : Ryzen 5 3600X,
six cœurs physiques, douze logiques. Benchmarks enfants avec BLAS/OMP/MKL limités
à un thread. Les sorties CPU restent des diagnostics de développement.

### Comptabilité et moteur

Le ledger monotone facture actions, constructions et 60 frames par restauration ;
préadmission des opérations complètes, allocation expérimentale 20 %, plafond 30 %.
Transport fermé, pixels/boutons seuls côté agent, aucune RAM dans les observations.
Une revue approfondie a corrigé l'ancien audit : 60 frames d'auto-détection par
loadROM étaient omises, et les resets naturels contiennent une boucle de mode
non bornée par l'API. Conserver les anciens résultats ; ils ne prouvent pas un
plafond strict. Erratum : `docs/ARC4_RESET_AUDIT.md`.

Nouveau moteur distinct `cold_emu_pool_v70.py` : instance ALE neuve par épisode,
ROM/binaire/options identiques, `system_random_seed=4753849`. Coût 132 frames,
128 pour Seaquest. Cette convention change la distribution des débuts d'épisode,
donc doit être commune aux futurs bras avant gel. Six jeux vérifiés, 36
constructions (7032 frames facturées), reprise non terminale inter-instance
identique sur 32 frames RGB brut/max/RAM (10132 facturées), 39 tests passent.
Cela reste une certification empirique, pas une preuve sur tous les états TIA.
Voir `docs/ARC4_COLD_START.md` et ses deux artefacts de certification.

### Perception et débit

- Pong brut et maximum de deux frames : calibrations réservées passées après
  scellement, erreurs initiales conservées. Voir `docs/ARC4_PONG_CALIBRATION.md`
  et `docs/ARC4_PONG_POOLED_CALIBRATION.md`. Anciennes captures : erratum de coût,
  géométrie inchangée. Extracteur de référence causal RGB seul, fond modal sans
  reset oracle. Optimisation exacte vérifiée par égalité complète des rapports.
- Chaîne Pong avec moteur froid : 16 × 2048 décisions, RGB IPC, référence,
  RAM évaluateur à chaque frame, branches et cibles C incluses : 1592,62
  décisions/s, 48 constructions, 170162 frames facturées. Sous cible provisoire
  2000 ; au-dessus du minimum technique 1000. ARC 3 tournait simultanément.
- Breakout brut : baseline négative conservée ; correction empirique de la balle
  scellée `6e48ef4` avant 17112, huit types passent sur 4096 frames réservées.
  Référence brute : rappel mobile 99,200 %, précision 72,388 % (2693 prédictions
  non appariées). Briques statiques incluses dans la précision, jamais mobiles
  du seul fait qu'un segment casse. `docs/ARC4_BREAKOUT_CALIBRATION.md`.
- Breakout temporel : protocole `10d88cc`, baseline 17121 passe sans correction,
  sceau `9dcb1b8`, validation 17122 passe aussi (2048 fenêtres, 9627 frames
  facturées). Boîtes raquette 99,414 %, balle 99,621 %. Revue indépendante close.
  **Le juge décrit les deux sources ; une occlusion de même rouge peut rendre
  l'identité indécidable depuis le seul maxRGB.** Garder cette portée dans toute
  mesure de perception. `docs/ARC4_BREAKOUT_POOLED_CALIBRATION.md`.
- Chaîne Breakout : baseline 642,61/s, intervalles équivalents 803,66/s,
  backend de calcul RGB compilé 1005,41/s. Les 16 rapports restent exactement
  identiques, 186450 frames facturées. Cible 2000 non atteinte. Backend optionnel
  `native_reference_v70.py`, C/DLL x64 sans imports avec empreintes vérifiées,
  revue Meitner close, 18 tests natif/référence passent. Aucun changement aux
  décodeurs scellés. Voir `docs/ARC4_NATIVE_REFERENCE.md`. Contrôle Pong natif
  ultérieur 1355,54/s, inférieur au NumPy antérieur sous charge variable, mêmes
  16 rapports : ne pas affirmer de gain contrôlé ou universel.
- Freeway brut : baseline négative `fc180c0`, V1 encore négative `45c78e3`,
  V2 corrige checker poulet/ligne jaune, bouclage/clip visible voitures et phase
  empirique voie4. Critères inchangés, revue Ohm close. Sources finales et dix
  empreintes commitées `b9d5a2b` avant17212. Échec ponctuel d'index Git sur le
  sceau JSON ; capture poursuivie, métadonnées commitées ensuite `92c0a42`.
  Déviation documentée, ne pas présenter ce dernier commit comme antérieur à
  l'ouverture. Candidat inchangé et hash antérieur vérifié. 17212 passe12types,
  présence joueur99,683%, boîtes99,633%,4228frames facturées. Trajectoires voitures
  identiques entre seeds ; leur répétition n'est pas une validation indépendante.
  Voir `docs/ARC4_FREEWAY_CALIBRATION.md`.
- Freeway temporel : protocole/candidat fixé `ed7fd7a` avant17221, 2048 fenêtres,
  8323frames facturées, une seule construction, dernière répétition tronquée.
  Sceau commité `0c96911` avant17222 ; validation passe12types. Référence brute :
  rappel99,7407%, précision51,7902% seulement, joueur97,4016% ; erreurs conservées.
  Chaîne NumPy878,38/s, native1430,58/s, mêmes16rapports,165808frames facturées.
  Revue Meitner close ; dossier `v70_freeway_pooled/`, docs de calibration.
- Seaquest : protocole/candidat fixé `339704a` avant8192frames17311,6constructions,
  8960frames facturées. Baseline négative conservée `31752a2` : aucun type passe,
  sous-marins et missiles ennemis absents RAM ; bulles bleues confondues par le
  checker avec des missiles, dimensions joueur/plongeur approximatives. 17312
  reste fermée. Diagnostic sur développement en cours ; revue Ohm des limites
  et de la couverture supplémentaire. Aucun changement de seuil ni passe vide.
  Toujours séparer les appels dépendants et inspecter les codes de retour.

Les sceaux bruts/temporaux listent les SHA des sources : **ne pas modifier ces
fichiers**, y compris moteurs et décodeurs partagés, pour les optimisations.
Ajouter des adaptateurs distincts et conserver tous les résultats antérieurs.
La calibration brute scellée inclut neuf fichiers, la temporelle dix.
Les anciens fichiers Pong scellés restent aussi des références historiques.

### Suite autorisée

Terminer Seaquest (géométrie et couverture supplémentaires à définir avant capture),
puis calibrer MsPacman et Asteroids
sur développement avec protocole préalable, baseline conservée, revue, scellement
avant validation réservée. Aucun autre jeu de confirmation n'est ouvert.
Résoudre avant gate 0 : registre des seeds/jeux, budget d'entrée E1, couverture,
allocation expérimentale commune, budget proposé 12 GPU-h et débit CPU. La mesure
mort F1 sur cinq des six jeux est impossible telle quelle : Pong et Freeway
n'exposent pas de vies ALE ; fixer les événements éligibles avant gel, aucun
F1 parfait attribué à un jeu sans positifs. Définir plus tard, avant les gels
correspondants, l'estimateur séparant avance initiale et adaptation accélérée.
RAM et référence sont des instruments externes, jamais des labels gratuits pour
l'agent apprenant. Les primitives apprises doivent provenir de ses expériences.

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
