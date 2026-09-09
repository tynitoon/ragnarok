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

**État actuel, 9 septembre à 11:37 : confirmatoire REPRISE et active.** Revue
indépendante close, huit tests passent, preuves et lanceur committés `b815be2`.
Dry-run complet conforme, puis contrôle des commandes et lancement à11:37:39
Paris : nouveau contrôleur PID20936, enfant s0 PID17340. Source active
`scripts/recover_confirm_v62.py` ; état `v62_confirm_recovery_state.json` =
`running/confirm_s0`. Logs `v62_confirm_recovery_stdout.log` / stderr, progression
expérimentale dans `v61_confirm_s0.log` puis s1/s2. Vérifier ces fichiers et les
commandes réelles, jamais les seuls numéros de PID (Windows les réutilise).

La première tentative a été interrompue par Windows Update. Les anciens
processus18416/8368 ont disparu. `v62_confirm_state.json` est le relevé historique resté `running`,
pas la preuve d'un processus vivant. L'événement système1074 à06:30:31 annonce
le redémarrage planifié ; dernier boot06:39:14.502. Deux objectifs M4 du premier
monde sont au log, mais le répertoire des checkpoints de bras est vide : le bras
doit être rejoué avec les mêmes graines, sans sélectionner selon ces résultats.
Copies des quatre fichiers et événements : `v62_confirm_interruption_20260909/`.
Son `budget_correction.json` recalcule correctement les ISO UTC :2166s facturées,
84742,0902837s restantes, projection totale34,3461h. Le champ9366s du premier
`interruption.json` est une erreur PowerShell préservée, pas le budget retenu.
Le lanceur distinct `scripts/recover_confirm_v62.py` ne modifie aucun ancien état ni lanceur,
préserve le préfixe du journal gelé en append et utilisera
`v62_confirm_recovery_state.json` / stdout / stderr distincts. Son lancement
est confirmé ci-dessus. Ne modifier aucun lanceur, preuve, état antérieur ou
source gelée pendant son exécution ; ne jamais lancer un deuxième calcul.

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

Dernier relevé du 9 septembre vers03:26 : lignées0 et1 terminées, quatre mondes
M1–M4 chacune, sorties et empreintes vérifiées. Durées contrôleur8100,22s puis
7568,43s (logs8090s/7560s). Contrôleur29812 actif, prétrain de la lignée2 PID22532
en cours. Projection actualisée≈34,21h sous42h, en estimant la dernière lignée
avec la plus lente des deux mesures. B_test6/N9/G'2 conservés ; pas de probe ni
de confirmatoire. Reçus `v62_pretrain_s0_receipt.json` et `v62_pretrain_s1_receipt.json`.
Revue indépendante Meitner close : cinq sorties et 17 sources gelées vérifiées,
checkpoints M1–M4 intègres ; couverture finale exacte39/44, courbes complètes.
Lire les logs réels `v61_pre_s0.log` puis s1/s2 :
les sorties des enfants ne remontent pas dans le log stdout du contrôleur Windows.
Relevé vers04:37 : lignée2 M2 terminé (aire0,630, couverture0,545), M3 en cours ;
premier objectif M3 terminé à4105s cumulées. Toujours aucun probe ni confirmatoire.

Relevé vers05:29 : les trois lignées sont complètes. Lignée2 vérifiée
indépendamment : quatre mondes,16courbes, checkpoints intègres,17sources gelées
inchangées.7147s au log,7155,080152s contrôleur ; couverture finale35/44=79,5455 %.
Projection totale33,9425h, marge8,0575h ; B6/N9/G'2 conservés. Reçu
`v62_pretrain_s2_receipt.json`. Le contrôleur29812 a enchaîné le probe, enfant21096,
état `running/probe`, départ à22823,7297333s cumulées. Vérifier l'état avant toute
action. Il s'arrêtera après le probe pour revue, aucun confirmatoire lancé.

Probe maintenant terminé et revu : `v61_probe.json`, SHA244f9ee0…, reçu indépendant
`v62_probe_review.json`. M4=618/768=80,4688 %, R=398/768=51,8229 % ; delta0,286458,
3SE pooled0,072446, PASS. Les12AUC M1–M4 ont été reproduites exactement en NumPy
depuis les checkpoints. M3 n=123 seulement : aucune accumulation monotone à
affirmer ; AUC M4 exercée0,6683 n'est pas une table parfaitement apprise.
Probe1087,029983s ; dépense totale17,8589h, projection33,7444h, marge8,2556h.
Contrôleur initial arrêté, statut `probe_passed_review_required`.
Nouveau lanceur séparé `confirm_resume_v62.py` préparé, revues Meitner/Ohm en
clôture, sans modification du code gelé. Il exige le reçu committé, protège les
sorties achevées et checkpoints entre enfants, applique G puis B si nécessaire,
et s'arrête après trois lignées complètes pour revue. Ne lancer qu'après commits,
dry-run et contrôle des processus ; aucun confirmatoire encore lancé à cette note.

Mise à jour du 9 septembre à 06:03 : reçu, lanceur et tests committés `998c43b`.
Les deux revues sont closes, 15 tests CPU passent. Dry-run conforme B6/N9/G'2,
86908,0902837 s disponibles ; aucun processus ARC 3 antérieur actif. Lancement
confirmé à 06:03:08 Paris : contrôleur PID18416, enfant s0 PID8368. État
`craft_v6_out/v62_confirm_state.json` = `running/confirm_s0`. Logs contrôleur
`v62_confirm_stdout.log` / `v62_confirm_stderr.log`, progression réelle dans
`v61_confirm_s0.log`, puis s1/s2. Projection environ 15,89 h ; vérifier état,
PID et commande réels avant toute action. Le contrôleur s'arrête pour revue
après les trois lignées. Ne modifier aucune des 17 sources gelées, aucun des
deux lanceurs, ni les artefacts/états antérieurs référencés par les empreintes.

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
- Seaquest (9 septembre, vers04:47) : baselines et corrections V1–V4 conservées.
  Collecte V4 `06caf0a` commitée avant17361 :32768images/33280frames facturées,
  quatre constructions, enfin patrouille de surface visible sur4788images.
  V4 corrige orientation des missiles bleus, occultations jaunes, supports de
  surface et fragments gris. Revues indépendantes Ohm/Meitner consignées en JSON.
  La collection17361 passe les sept catégories/sous-type, mais17341 reste négative
  pour joueur (hauteur6 ignorée) et sous-marins (fragments immergés séparés).
  V5 minimale : joueur h>1, regroupement gris immergé8x11 ; six témoins audités.
  Les cinq développements passent leur géométrie observée séparément ;17361
  passe toutes catégories/surface. Revue Meitner finale close et reçue en JSON.
 32tests CPU ciblés passent par groupes. Contacts jaunes larges et bas restent
  une limite déclarée, les catégories absentes restent FAIL/null.
  Nouveau `validate_seaquest_reserved_v70.py` : futur couple fixe17312/17362,
  sources puis sceau réellement commitées avant exécution. Guard Git5tests,
  orchestration8tests dont capture simulée→vrai lecteur ; revue close après
  correction du nom de format historique. Sources commitées126a2eb, sceau et
  revue commitées23cf31d AVANT ouverture. Exécution du couple réservé lancée
  vers04:47 : `python -u -m scripts.validate_seaquest_reserved_v70 run`.
  Les25sources du sceau sont désormais figées, y compris le guard générique.
  Ne pas les modifier. Résultats terminés vers04:52 : les deux séquences PASS.
 17312 :8192images/8704facturées,4loads,six catégories couvertes et calibrées.
 17362 :32768images/33280facturées,4loads,surface3463appariements within1.
  `reserved_raw_result.json`, `reserved_coverage_result.json`, `reserved_pair_result.json`.
  Revue indépendante Meitner close, reçu commité1e3d8b5 :160archives,
 40960images/41984frames facturées, chronologie et25sources confirmées.
  Module temporel séparé `pooled_seaquest_v70.py`, protocole17321/17322,
 8192décisions ; revue Ohm close avant capture, quatre tests module et deux
  tests captureur passent. Commit2b03bd9 avant17321. Baseline temporelle PASS :
 8192décisions/32759frames ordinaires+512froides=33271facturées,4constructions.
  Six catégories et sous-type surface passent, tous les appariements within1,
  surface1029appariements. Revue Ohm close, source33 e911d36 et sceau f62333f
  committés avant17322. Réservé terminé PASS :8192décisions/33396facturées,
 5constructions,surface159matches exacts,joueur3/missilejoueur5erreurs within1
  conservées. Ne modifier aucune des33sources. Revue Meitner du résultat en cours.
  Référence brute nouvelle capture uniforme17371,protocole81a7fa4 avant capture,
  résultat3698beb : rappel93,6393 %, précision25,4255 % seulement, joueur73,1446 %.
  Sous-marins/missiles ennemis absents=null.8960facturées. Ne jamais entraîner
  la référence sur les captures réservées/couverture qui l'interdisent.
  Adaptateurs graphe/scoreur/worker distincts revus puis committés2cb7b29.
  Égalité sur2x8192fenêtres, raccord réel3582facturées, reçus a025cb9.
  Chaîne53sources,16x2048 : NumPy999,8425/s,natif1311,5635/s,174380facturées
  chacun. Précision27,8876 %, rappel composants mobiles93,4980 %, non comparable
  directement au rappel des acteurs bruts. Revue Ohm close :16rapports complets
  exactement identiques,53sources et build/DLL vérifiés, reçu cpu_backend_equivalence.json.
  Voir `docs/ARC4_SEAQUEST_CPU.md`. Ms. Pac-Man : scope5catégories, protocole
  etbaseline nominale revus/committés9714014 avant17411 (cléROM ms_pacman).
  Capture8192images/8720facturées,4constructions. Baseline V0 FAIL cinq catégories,
  préservée c8e64ef : joueur géométrie36,72 %, fantômes présence11,96 %, fruit
  absent/null, grosses pastilles confondues avec décor, petites pastilles présence49,60 %.
  Réservé17412 fermé. Trois diagnostics nouveaux : joueur (pas de correction simple
  x/largeur), fantômes (RAM précédente pour position, sélection de rendu à établir),
  grosses pastilles (flags117 et phase30&8 courants exacts sur32768supports,337occultations).
  Aucun correctif V1 appliqué. Voir `docs/ARC4_MSPACMAN_CALIBRATION.md`.
  Ne pas relancer si une capture/sortie existe, même incomplète.17332/17342/17352
  restent retirés sans ouverture. Tous les anciens échecs restent négatifs.
  Voir `docs/ARC4_SEAQUEST_CALIBRATION.md`. Toujours séparer les appels Git
  dépendants et inspecter les codes de retour avant toute capture.

Les sceaux bruts/temporaux listent les SHA des sources : **ne pas modifier ces
fichiers**, y compris moteurs et décodeurs partagés, pour les optimisations.
Ajouter des adaptateurs distincts et conserver tous les résultats antérieurs.
La calibration brute scellée inclut neuf fichiers, la temporelle dix.
Les anciens fichiers Pong scellés restent aussi des références historiques.

### Suite autorisée

Seaquest est terminé pour brut/temporel, référence et débit CPU ; Pong, Breakout
et Freeway également calibrés aux étapes décrites plus haut. Aucun apprentissage
ARC4 ni gate lancé. Les contrôles RAM/palettes restent exclusivement externes.

**Ms. Pac-Man : calibration brute réservée complète et revue.**
Développements17411/17431 et tous les échecs antérieurs conservés. Source
réservée5a2ee619 puis scellé1fddd2c (31sources+7métadonnées), contrôles réels
avant/après conformes. Réservés17412uniforme :8192images/8852facturées/5loads ;
17432couverture :32768images/33824facturées/8loads. Les quatre catégories
observées passent l'uniforme, fruit strictement absent/null/FAIL ; les cinq
passent la couverture. `raw_reserved_pair_v3.json` pair_pass=true, revue
`reserved_pair_review_v3.json` SHAa53c16554af0bc91a57ea3876aabb2a26e2323e78793b96b10e29a91fe99ef89.
Fantômes couverture31682/31914comptages positifs,85316/85854within1 ; petites
32464/32504comptages,2517375/2517439within1.160archives/40960lignes de charges
vérifiées indépendamment. Ne modifier aucune source scellée ou preuve ; aucun
nouveau réglage à partir des réservés. Voir `docs/ARC4_MSPACMAN_CALIBRATION.md`.
Suite : calibration des fenêtres réellement reçues avec un nouvel adaptateur,
puis référence/débit. Trois RAM adjacentes sont nécessaires pour les deux RGB.
`triple_ram_tap_v70.py` et six tests CPU préparés ; aucun émulateur temporel ni
seed17421/17422 ouvert. Protocole et sources à figer avant ce développement.

**Asteroids : réservé17512 terminé PASS, revue indépendante en cours.**
Développement17511 :8192images/8852facturées/5loads, baseline et corrections
négatives conservées. Le premier PASS joueur/tirs hors ROI est déclaré non
conforme ; la composition finale respecte [16,194). Rochers V5 :4065/4096
comptages positifs,26628/26673within1. Pièces visibles par allocation, aucune
fusion d'allocations RAM ni prétention d'identités physiques. Les glyphes et
chronologies sont des hypothèses externes, jamais des entrées pour l'agent.
Composition et lanceur revus ;31sources du commit73fdf4c, scellé committé158a976,
SHA7b410031334df36aa47c0af1aa964a98f400d487b3ee13ae36f2d55a4b3f713f.
Préflight réel PASS puis capture17512 à11:58:47Z, fin du scoring12:01:34Z.
8192images/8852facturées/5loads, joueur2051boîtes exactes, tirs918exacts,
rochers4073/4096comptages positifs et27541/27577within1.23images de comptage
en erreur gardées. `reserved_visible_v1.json`, SHA
d24fe0badfa72fb42601cb7e0e878251ad41db589344bfecc5b32e817408faf9,
tentative=complete. Revue scientifique bornée confiée à review_recovery_v62,
reçu demandé `visible_reserved_review.json`. Toutes les sources/preuves sont
figées ; ne pas relancer17512 ni régler ses erreurs. Voir le détail dans
`docs/ARC4_ASTEROIDS_CALIBRATION.md`. Aucun autre jeu de confirmation ouvert.
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
