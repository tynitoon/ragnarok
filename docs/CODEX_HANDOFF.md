# Ragnarok — point de reprise

19 octobre 2026. **La pause du 10 septembre est levée** : Jérémie a demandé de
reprendre, de revérifier l'état et de redéfinir les tâches. L'ancien point de
reprise, intact, est dans `docs/history/CODEX_HANDOFF_20260910_pause.md`.
Aucun processus d'expérience actif au contrôle (zéro `python.exe`).

## Ce qui est établi

- **ARC 3 v62 : INCONCLUSIVE** (`docs/ARC3_V62_CONFIRMATION_RESULTS.md`).
  Δ +0,0549 contre 2 se 0,0528, mais deux lignées sur trois non positives.
  Par position : palier 4 +0,161 (se 0,015), palier 3 +0,108 (sous le seuil de
  formulation 0,149), palier 2 **−0,104** (l'expérience gêne sur l'objectif
  facile). Sonde de connaissances positive (618/768 contre 398/768). Verdict
  gelé, ne pas recalculer. C'est le signal le plus net de l'histoire du projet,
  mais ce n'est pas une démonstration d'apprentissage accéléré.
- **ARC 4 : aucun agent ne joue encore.** Du 8 au 10 septembre : 432 commits,
  163 documents, 137 scripts `_v70`, presque tous des micro-diagnostics de
  perception (cinq ajustements négatifs, un positif limité à une famille
  artificielle). Le dernier candidat (encodeur photométrique de 1 712 paramètres
  sur images 8×8 synthétiques) s'est arrêté à une garde CPU avant tout calcul.

## Constat de la reprise

Le dispositif de mesure a dévoré le projet : reçus SHA, fsync, préflights et
revues pour des fits de quelques secondes. C'est exactement le risque R15 que
le projet avait lui-même nommé. La rigueur reste obligatoire là où un résultat
est revendiqué ; elle n'a pas à précéder chaque ligne de code.

## Décisions prises à la reprise

1. **Exception de relance photométrique : non retenue.** Le STOP gelé reste
   tel quel ; la proposition et son préflight restent archivés. Cette ligne de
   diagnostic est close, car même réussie elle ne rapprochait pas d'un agent qui
   joue. Aucun critère n'est modifié par cette décision.
2. **Rigueur proportionnée.** Pré-enregistrement, gel et revue contradictoire
   seulement aux jalons qui portent une revendication (gate, confirmation). Le
   code d'ingénierie, les prototypes et les essais exploratoires se font vite,
   étiquetés « exploratoire », sans valeur de preuve.
3. **Branche pré-déclarée d'ARC4_VISION §5 invoquée pour avancer** : la
   perception apprise n'a pas passé ses diagnostics ; l'extracteur classique
   devient un échafaudage **commun à tous les bras**, déclaré, et « apprend à
   voir » sort du plafond de revendication tant que la perception apprise ne
   passe pas un gate 0-B formel. La perception apprise reste une piste
   parallèle, pas un préalable.

## Objectifs redéfinis

**But inchangé** : un agent qui apprend des jeux depuis l'écran et les boutons,
Atari → NES → 16 bits → PC, et qui apprend chaque génération plus vite qu'un
agent neuf. Une avance au départ ne prouve pas un apprentissage plus rapide.

**Jalon A — tranche verticale Atari (ingénierie, exploratoire).** Un agent
complet qui joue en direct sur 3 jeux (Pong, Breakout, Freeway) : extracteur
commun → moi par contrôlabilité → découverte des boutons → lecteur de score et
détecteur de mort → pulsions et arbitre → politique par auto-imitation
rétrospective → registre de contacts. Livrable : une vidéo, des courbes, et le
nombre d'images nécessaire pour dépasser le hasard. Coût visé : CPU surtout,
quelques GPU-h. Aucune revendication de transfert.

**Jalon B — gate 0-A formel** (instrument : moi, lecteur, mort, boutons, débit)
sur la tranche du jalon A, gelé et revu une fois.

**Jalon C — première mesure de transfert** dans Atari selon ARC4_VISION §4
(gate WORTH K* avant toute lignée), puis entrée NES.

**Option ARC 3 v63 (à décider par Jérémie, GPU en parallèle).** Tester sur des
mondes neufs l'hypothèse née de v62 : l'avantage se concentre sur les paliers
difficiles et l'expérience nuit sur le facile. Nouveau pré-enregistrement,
nouveaux mondes 8330+, étiqueté « hypothèse issue de v62 ». Pour détecter
l'effet du palier 3 (+0,108) il faut environ 17 mondes, soit ~35 GPU-h.

## Contraintes inchangées

Pas de RAM ni d'oracle dans les observations ; RAM réservée à l'évaluation
externe. Pas de poids préentraînés téléchargés. Jeux réservés jamais ouverts :
Boxing, Frostbite, Riverraid, Kangaroo. Python principal
`C:/Python314/python.exe` (Torch) ; `.venv-arc4` pour ALE sans Torch (SETUP.md).
Ne rien supprimer sans accord. Commits ciblés. L'automation « Suivi Ragnarok
ARC 4 » reste en pause.

## État au 19 octobre, pause demandée par Jérémie

- Analyse v62 faite : au palier 3, écart-type par unité 0,46 pour un effet de
  0,11 ; une v63 confirmatoire demanderait ~70 mondes (~140 GPU-h). Pas de v63
  pour l'instant ; piste utile = diagnostiquer pourquoi la moitié des
  apprentissages calent (aucune réussite, donc aucune ligne d'auto-imitation).
- Jalon A : `.venv-arc4` + ale-py 0.12.1 OK ; Breakout 7 100, Freeway 4 700,
  Pong 9 500 images/s par cœur ; restauration de sauvegarde ~0,5 ms. Jeux de la
  tranche : Breakout, Freeway, Space Invaders (Pong optionnel : score à deux camps).
- Prochaine étape : inventaire du code `_v70` réutilisable (agent arrêté avant
  rapport), puis construction de la tranche verticale.

## Jalon A — avancement au 20 octobre (exploratoire, aucune revendication)

Code neuf et compact : `ragnarok/agent_v71/` (env, me_buttons, hud, tracker, perceiver,
explore, scorer côté évaluateur) et `scripts/slice_*_v71.py`.
- **Étape 1 OK** (895e64b) : « moi » et effets des boutons par branches de sauvegarde.
  Corrélation avec la RAM 0,98-1,00 sur Breakout, Freeway, Space Invaders, Pong ;
  sens des effets 100 %. Défaut : FIRE mal classé (devrait être « apparition »).
- **Étape 2 partielle** (0bbd1b5) : compteurs sans OCR. Breakout score 104/104
  (précision et rappel 1,00), vies précision 1,00 rappel 0,80. Échecs : Space
  Invaders (chiffres fragmentés ou soudés aux aliens), Pong (score adverse choisi).
- **Étape 3 en échec** : exploration par nouveauté avec sauvegardes. Cellules
  Go-Explore (11×8) : 1 seule cellule sur Freeway, pire que le hasard. Cellules
  (position de moi 16×16, états des compteurs) : Freeway score 4 contre 12 pour le
  hasard avec répétition d'actions, au même nombre d'images ; 7 611 cellules (des
  régions non-compteurs entrent encore dans la clé) ; ~300 images/s, 20× trop lent.
- **Suite** : (1) profiler la boucle d'exploration ; (2) clé de cellule = position
  de moi seule + compteurs validés hors ligne sur l'étape 2, pas réajustés en ligne ;
  (3) si l'exploration ne bat pas le hasard répété sur Freeway et Breakout à budget
  égal, l'abandonner au profit d'une boucle « jouer, rejouer les réussites »
  (auto-imitation rétrospective) partant du hasard répété.

## Jalon A — 21 octobre : premier agent qui joue (Freeway)

Chaîne complète, écran et boutons seulement, aucune récompense, RAM côté évaluateur :
moi par branches de sauvegarde → compteurs (score/vies) par forme → exploration
(nouveauté × pulsion de score, veto de mort) → démonstration → clonage puis
auto-imitation « backward » des réussites plus rapides que la médiane → jeu EN DIRECT.

| Jeu | Explorateur (meilleur score) | Hasard répété, mêmes images | Politique en direct* | Hasard répété en direct* |
|---|---|---|---|---|
| Freeway | 25 | 12 | **21,6** (20-23) | 5,6 |
| Breakout | 16 | 14 | 2,4 | 2,0 |

*graine 7, actions collantes 0,25, sans sauvegarde, 5 parties. Référence humaine
Freeway (Atari-100k) 29,6 → ~0,73 humain. Breakout : échec, la vue 64×64 centrée sur
la raquette ne voit sans doute pas la balle assez tôt ; démonstration faible (10 pts).
Correctifs décisifs du jour : le suivi de « moi » (couleurs propres + fond réparé sous
l'agent, corrélation RAM 1,000 en direct) ; avant, la même politique faisait 5,6.
Space Invaders et Pong restent exclus (lecteur de score).

### Recette unique v1 (21 octobre)

Même code pour tous les jeux : départs tirés au hasard le long de la démonstration ;
chaque événement de score atteint est une leçon (le segment qui y mène) ; on imite les
segments plus rapides que la médiane du tour ; chaque mort déclenche « une mort, une
leçon » (rembobinage 8/16/32 pas, la branche qui survit est une démonstration
d'évitement) ; vue = fenêtre ego + vue globale, chacune avec l'image précédente.
En direct (graine 7, collant 0,25, sans sauvegarde, 5 parties) :
Freeway **22,4** (20-25) vs hasard répété 5,6 ; Breakout **5,0** vs 2,4.
Essais intermédiaires consignés dans `craft_v6_out/v71_play_*_v*.log` (Freeway 21,6 /
18,8 / 16,8 / 16,6 / 14,6 selon les variantes ; Breakout 2,4 / 3,8 / 3,0 / 1,8 / 5,0).

### Jalon A tenu sur 6 jeux ; pilote du jalon C lancé (22 octobre)

Apprenant unifié `ragnarok/agent_v71/learner.py` (politique en EFFETS canoniques, adaptateur
effet→bouton par jeu). En direct contre hasard répété (5 parties, graine 7, collant 0,25) :
freeway 20,2/4,6, carnival 620/224, demon_attack 469/74, name_this_game 1894/744,
breakout 4,6/1,8, assault 340/231 ; seaquest 112/92 et chopper_command 360/420 ne battent
pas le hasard. Recensement 40 jeux : lecteur de score fiable sur 11 (`v71_survey.json`).

Pilote C (exploratoire, aucune revendication) : `scripts/slice_transfer_v71.py`, sorties par
essai dans `craft_v6_out/v71_transfer/` (reprise automatique). Famille des tirs verticaux
(assault, carnival, demon_attack, name_this_game), un jeu laissé de côté à tour de rôle ;
agent E entraîné en séquence sur les 3 autres (poids gardés, exploration guidée 50 %) contre
deux agents neufs Fa/Fb ; budgets 0/100/300/1000 itérations d'exploration. Si le pilote montre
un écart au-delà de |Fa−Fb|, pré-enregistrer une confirmation sur des jeux neufs.

### CORRECTION (21 octobre) — la référence « meilleur bouton constant »

Presser un seul bouton toute la partie (graine 7, collant 0,25) donne : freeway UP 21,0 ;
assault UP 476 ; demon_attack FIRE 480 ; name_this_game FIRE 2770 ; seaquest 180 ;
chopper_command 1167 ; breakout 0 ; carnival 0 (`v71_constant_baseline.json`).
**L'agent ne bat le meilleur bouton constant que sur Breakout (4,6) et Carnival (620).**
Sur Freeway il a découvert seul « monter » (≈ le bouton constant), pas davantage ; sur les
autres jeux il fait moins bien que le meilleur bouton. Les formulations antérieures
(« bat le hasard sur 6 jeux sur 8 ») étaient exactes mais trompeuses sans cette référence ;
cette référence s'ajoute désormais à toute évaluation. Name This Game : tous les bras du
pilote font exactement 1894 → la politique n'influençait pas le jeu (résultat retiré).

Pilote de transfert (`craft_v6_out/v71_transfer/`) : au départ (budget 0), l'agent
expérimenté est bien pire que l'agent neuf (assault 17 vs ~190, carnival 116 vs ~600,
demon_attack 22 vs ~190) : transfert NÉGATIF ; ensuite aucun écart au-delà de |Fa−Fb|.

### 21 octobre (après-midi) — diagnostic appliqué, v72 → v75 (exploratoire)

Code : `ragnarok/agent_v71/` sur les branches `arc4-v73` (`../ragnarok_v73`), `arc4-v74`
(`../ragnarok_v74`), `arc4-v75-offline` (`../ragnarok_v75`), non encore fusionnées dans master.
Protocoles figés avant lancement : `docs/ARC4_V72_PROTOCOL.md` (master), `ARC4_V73_PROTOCOL.md`,
`ARC4_V74_PROTOCOL.md`, `ARC4_V75_OFFLINE.md` (sur leurs branches). Référence partout : le meilleur
bouton constant sur les mêmes parties ; critère : le dépasser de plus de 2 erreurs types.

- **v72** (corrections D1-D5, imitation de branches tenues 32 pas) : au niveau du constant au mieux.
  Assault 382/365 contre UP 400 ± 25 ; Freeway 19,9/21,1 contre 21,2 ; Demon Attack 245/356 contre
  FIRE 526 ; Name This Game 2345/2502 contre FIRE 2770 (identique à chaque partie). Carnival et
  Breakout à compléter (deux jobs sautés par la garde RAM : relancer `python -m scripts.slice_v72 all`).
- **Signal de mort lu à l'écran** (compteur de vies OU « disparu 2 pas », coupé si l'agent se perd
  > 20 %) : précision 0,97-1,00, rappel 0,73-1,00 sur 4 jeux, aucune fausse alerte sur Freeway et
  Carnival (`craft_v6_out/v73/death_check.json`). Le suivi de soi de Name This Game était perdu
  100 % du temps (amorçage) : corrigé.
- **v73 interrompue** : l'explorateur comptait « disparu » par une autre voie que le signal validé
  (démo Freeway vide). Assault seul complet : 403/408 contre 407 (au niveau du constant).
- **Constats de développement** : le gain mesuré par branches est réel (pas du bruit de sélection) ;
  le goulot est la distillation en politique réactive. Hors ligne honnête (panel du 21 octobre), la
  part qui dépend de l'écran est faible sur Assault (~10 % de l'oracle ; deux tiers du gain v74 étaient
  un biais « une pause ») et nulle sur Demon Attack, dont l'oracle ressemble à du hasard. Défaut
  d'entrée trouvé : la jauge de chaleur d'Assault est hors de la vue globale. Transfert hors ligne
  Assault → Demon Attack : nul (frais = transporté = placebo).
- **v74 en cours** (amélioration par simulation sur la base constante de l'agent, DAgger, décisions
  tenues 8 pas ; graines neuves 4/5, évaluation 13/14 ; critère par graine). Premiers chiffres
  Assault graine 4 : 387 ± 34 et 421 ± 33 (192 racines) contre 383 ± 22. Files : `slice_v74 all 0/2`
  et `1/2` dans `../ragnarok_v74`, reprise automatique.
- **v75 hors ligne prêt** (T1-T3, seuils écrits) dans `../ragnarok_v75` ; à lancer quand la RAM le
  permet (un processus à la fois).
- Ressources : superviseur RAM actif ; entre 13 h 47 et au moins 14 h 45, tous les calculs Ragnarok
  sont restés suspendus (Tavern) ; les gardes de lancement attendent désormais au lieu de sauter.

### Reprise après le redémarrage du PC (21 octobre, ~17 h)
Tous les calculs étaient arrêtés (demande de Tavern) ; rien n'est perdu, les files reprennent sur
leurs sorties. Jérémie a agrandi le fichier d'échange (la limite était la mémoire réservée). Ordre
de relance (depuis chaque copie, en arrière-plan ; skill `ragnarok-run`) :
1. superviseur : `C:/Python314/python.exe scripts/ram_supervisor.py --low 12 --high 14 --clow 12 --chigh 14` (dans `F:/dev/ragnarok`) ;
2. v80, nouvelle voie (agent qui se fait une théorie du jeu) : `python -m scripts.run_v80_dev freeway assault` dans `F:/dev/ragnarok_v80` (conception `docs/ARC4_V80_DESIGN.md`, revue appliquée) ;
3. v74 figée : `python -m scripts.slice_v74 all 0/2` et `all 1/2` dans `F:/dev/ragnarok_v74` ;
4. v72 : `python -m scripts.slice_v72 all` dans `F:/dev/ragnarok` (trous : Carnival, Breakout).
Toutes les gardes attendent 14 Go de RAM ET de mémoire réservée ; au plus 2-3 processus à la fois.

### 22 octobre (nuit) — v80 « l'agent qui se fait une théorie du jeu » : milestone 1 négatif
Branche `arc4-v80-theory` (`../ragnarok_v80`), conception et journal complets dans
`docs/ARC4_V80_DESIGN.md` (§8-13). L'agent découpe l'écran en objets, fait des expériences par
branches sœurs, écrit une théorie lisible (types, mouvements, mes tirs, contacts, jauges, zones,
rebouclage, replacement, présages) et imagine la suite. Graines de développement 98/99.
- G0 perception : réussi (Freeway 99,9 % des voitures ; Assault 1,7 % de pas mauvais).
- Théorie lisible et juste : Freeway (voitures qui rebouclent, zone de points en haut, replacement
  en bas, « une voiture me repousse ») ; Assault (je tire vers le haut à 16 px/pas, la jauge de chaleur
  monte quand je tire, la surchauffe tue).
- G1 Freeway réussi à l'essai 4 mais NON confirmé sur la graine 99 (le test était de plus identique
  entre graines : corrigé). G1 Assault : critère d'arrêt KC1 atteint (classement des morts 0,28/0,47,
  prédiction pire qu'une moyenne). G2 jamais atteint. Ligne « imagination » arrêtée sur Assault.
- Causes : couche d'objets bruitée (le fond absorbe les jauges ; débris d'explosion pris pour des
  présages ; types fragmentés par la forme ; une absence à l'écran est ambiguë).
v74 : Assault négatif au critère par graine (387/471 et 421/373 contre 383) ; Demon Attack négatif
(146-149 contre 574) ; file en pause. Superviseur RAM + mémoire réservée actif ; la machine est
très chargée par Tavern la nuit.

### 22 octobre (3 h) — v80 clos ; v81 « atlas d'entraînement » ouvert
Lecture honnête de v80 dans `../ragnarok_v80/docs/ARC4_V80_DESIGN.md` §14 (types de notions écrits par
moi ; KC1 Assault = couche d'objets ; Freeway essai 4 indécidable ; erreur de méthode reconnue).
v81 (`../ragnarok_v81`, branche `arc4-v81-atlas`, protocole `docs/ARC4_V81_ATLAS.md` figé) : mesurer
d'abord si « plus vite » est visible (G-place : coût d'entraînement d'un agent neuf jusqu'à des paliers
de score), atlas « où je suis × ce que mes actions font là », a priori d'affordance testé sur des cases
inédites ; jeux candidats Bank Heist, Wizard of Wor, Amidar, Crazy Climber (Freeway en témoin) ;
graines 8101-8106 ; 7 jours au plus ; chances annoncées ≈ 0,2 pour l'étape entière.

### 22 octobre (~4 h) — v81 arrêtée par KC-pool ; le premier obstacle est la perception générale
E sur 5 jeux (graine 8101) : seul Amidar éligible (et sa mort n'est vue qu'une fois sur 47). Sur Bank
Heist, Crazy Climber et Wizard of Wor l'agent lit une autre zone que le score (détecteur de compteurs
dépendant du hasard de l'échauffement) et ne voit pas ses morts. Détails : `../ragnarok_v81/docs/ARC4_V81_ATLAS.md` §9.
Proposition : étape « perception générale » (moi, score, mort) mesurée contre la RAM sur un large lot,
avec seuils fixés d'avance, avant toute nouvelle mesure d'apprentissage.

### 22 octobre (~5 h) — DÉCISION DE JÉRÉMIE : « sens du score et des vies »
Choix explicite de Jérémie (question posée après v81) : l'agent reçoit le score (ses hausses) et le
nombre de vies comme des sens, l'équivalent d'un humain qui lit ces chiffres à l'écran. Tout le reste
(moi, effets des boutons, objets, dangers, valeur) reste appris depuis les pixels et les boutons. La
RAM reste interdite à l'agent (évaluation seulement). Motif : l'innovation recherchée est l'apprentissage
qui s'accumule ; la lecture générique des chiffres (étape v82 « perception générale », verdict du juge
gardé dans `../ragnarok_v81/docs/panels/v82_perception_judge.md`) devient un chantier séparé, plus tard.
Tout résultat obtenu avec ces sens le dit explicitement.

### 22 octobre (~9 h) — v83 « carte des promesses » : protocole figé, code revu, A1 lancé
Worktree `../ragnarok_v83` (branche `arc4-v83-promise`), protocole figé `docs/ARC4_V83_PROTOCOL.md`,
synthèse chiffrée `docs/panels/v83_promise_map_synthesis.md`, écarts d'implémentation et corrections de
revue `docs/ARC4_V83_BUILD_NOTES.md`. Idée : l'agent apprend sur ses jeux d'avant « ce qui précède une
hausse du score / une perte de vie » autour de lui (deux têtes, BON et MAUVAIS) et s'en sert pour piloter
son exploration d'un jeu neuf ; contrôles : placebo à étiquettes décalées, bras neuf, bras figé.
- Sens : `ragnarok/evaluation/senses_v83.py` (hausses, vies, poignées opaques, 60 images par chargement,
  trace évaluateur rejouable). Le paquet agent `ragnarok/agent_v83` n'importe rien de l'évaluation (test AST).
- Revue contradictoire avant lancement (`wf_015d274f-fc1`) : 11 défauts réels corrigés avant toute image A1,
  dont des étiquettes sélectionnées sur leur issue et des crochets de pilotage absents.
- Rodage (graine 9130) : Amidar atteint 490 (barre 31) ; Freeway reste à 17 sous la barre 22,7 (UP constant).
- **A1 (G-place)** tourne depuis le worktree propre `../ragnarok_v83run` (détaché au commit `2b22e94`),
  2 tâches : `scripts/room_v83.py run 0/2` et `1/2` (reprise automatique ; sorties
  `../ragnarok_v83run/craft_v6_out/v83/{bars,room}`). Verdict : `scripts/room_verdict_v83.py` dans ce worktree,
  puis commit de `room_verdict.json` avant toute image des bras F, X, P, K*. Ne pas modifier ce worktree
  pendant A1 (provenance : code sale ou mêlé = Interruption).
- v83 s'est arrêtée à A1 (KC-place, 23 octobre) : les 24 jeux scellés et les 9 jeux du second tour n'ont
  pas été ouverts (aucune barre, aucun run). Ils restent utilisables par une version suivante sous son
  propre protocole.

### 23 octobre (~3 h) — v83 A1 : KC-place (négatif, gardé)
**A1 (G-place) : KC-place.** 6 des 35 jeux de développement ont de la place (alien, amidar, bank_heist,
ice_hockey, jamesbond, qbert ; seuil 13) ; calibration : breakout, ms_pacman. σ̂ = 1,08 ; se_proj = 0,48
(seuil 0,23). Le critère (b) échoue sur 29 des 35 jeux de développement (33 des 41 avec la calibration) et
écarte à lui seul chaque jeu non retenu ; (d) n'a écarté aucun jeu (calculé seulement si (b) passe).
Première notation : Interruption (5 des 6 rejeux Q*bert en écart, `room_verdict_first_pass.json`). Le
vérificateur a été corrigé côté évaluateur (un pas NOOP avant restauration, note 21), puis le verdict
recalculé sur les mêmes fichiers A1. Cela franchit la lettre du §6 (« défaut après notation jamais renoté
sur ce lot »), dans le sens défavorable à l'agent. Les comptes sont identiques dans les deux notations
(6/35, se_proj 0,48 ; sans Q*bert 5/35, se_proj 0,57) et les runs sont déterministes : aucune notation ni
reprise ne pouvait donner G-place. Revue du verdict : `wf_420b5f0a-bc1`. Détail et leçons :
`../ragnarok_v83/docs/ARC4_V83_A1_RESULT.md`. Rien n'a été mesuré sur « apprendre plus vite » (ni
promesses, ni transfert) ; les lots scellés restent fermés.

### 23 octobre (~5 h) — v84 « porter les questions, pas les réponses » : protocole figé
Panel de conception (5 propositions, 3 juges, synthèse, 3 audits ; `docs/panels/v84_*`), protocole figé
avant tout calcul dans le worktree `../ragnarok_v84` (branche `arc4-v84-clock`) :
`docs/ARC4_V84_PROTOCOL_EN.md` (fait foi, SHA-256 `bea9ed27…`) et `docs/ARC4_V84_PROTOCOL.md`.
Premier test G0 « horloge du savoir », sans image d'émulateur, sur les 82 journaux A1 : un a priori de
covariance à moyenne nulle appris sur 40 autres jeux fait-il apprendre plus vite (à expérience égale)
la prédiction BON/MAUVAIS d'un jeu tenu à l'écart, contre trois témoins réglés (F, P, U), avec contrôle
planté préalable ? En parallèle : explorateur V2 sans sondes, porte « place » G1b. Graines 9700-9749.
L'analyse exploratoire v83 (`../ragnarok_v83/craft_v6_out/v83/explore/`) reste scellée jusqu'au commit
de `g0_verdict.json`. Chances écrites : G0 ≈ 0,2 ; positif scellé ≈ 0,5 %.

### 23 octobre (~7 h) — v84 : G0 = RECIPE-INSENSITIVE ; V2 construit ; G1a lancé
- **G0 (horloge du savoir) : RECIPE-INSENSITIVE** (`../ragnarok_v84/craft_v6_out/v84/g0/g0_verdict.json`).
  Le contrôle planté positif a échoué deux fois (premier passage compté par prudence, puis l'unique
  modification de recette autorisée : seuil d'espace nul 1e-2) : X:C\* 1,07 puis 1,03 ; même le plafond
  intra-jeu K\*A n'atteint que 1,62 puis 1,44 (seuil 2). Le nul passe. Aucune courbe sur les vraies
  étiquettes n'a été calculée ; voie Σ fermée ; ce n'est pas un négatif sur le mécanisme. Leçon : cet
  apprenant logistique apprend déjà vite les conséquences ; la marge pour 2× est structurellement faible.
- **V2 construit** (`ragnarok/agent_v84/explorer2.py`) et testé (tests de construction 1-6 réussis ;
  test 7 : 0,45 Go, ~50 min par run de 4 M ; 3 tâches parallèles).
- **G1a** (exploratoire, 12 runs, graine 9710) tourne : `scripts/queue_v84.py g1a k/3` (processus
  détachés, journaux `craft_v6_out/v84/logs/`), rapport `scripts/room_verdict_v84.py g1a`. Ensuite G1b
  (porte « place », 41 jeux × graines 9712-9713, `queue_v84.py g1b`), puis verdict `room_verdict_v84.py g1b`.
  Ne pas committer de code pendant G1b (code mêlé = Interruption).
- L'analyse exploratoire v83 tourne encore ; son scellé est levable (g0_verdict committé) dès sa fin.

### 24 octobre (~9 h 40) — v84 : G1b = PROCEED (re-noté) ; G0-S = RECIPE-INSENSITIVE ; v84 s'arrête à la faisabilité
- **G1b : PROCEED, re-noté après correction du vérificateur de rejeu** (première notation gardée :
  Interruption, 2/200, Q\*bert ; notes 15-16 de `../ragnarok_v84/docs/ARC4_V84_BUILD_NOTES.md`).
  22/35 jeux de dev « de place », se_proj 0,142 (seuil 0,23), 21/35 atteignent M3 en ≤ 7 000 pas. Sans
  Q\*bert : 21/35, se_proj ≈ 0,15, 20/35. La légende « grimpe sur la plupart » repose sur (d) avec médiane
  = moyenne des deux graines ; lecture par graine : 18/35 et 17. Revue : `wf_f59985c8-809`.
- Établi : une recherche avec retours en arrière facturés (V2, sens du score et des vies) atteint M3
  (≈ 5×B′) sur 22/35 jeux de dev. Non établi : jeu en direct, apprentissage plus rapide grâce à d'autres
  jeux (rien n'est transporté), jeux neufs (jeux de dev déjà vus par A1), score vrai (hausses senties).
- Contre 6/35 pour A1 (v83) : indicatif seulement (plafond 2 M contre 4 M, échelle, barres, (c) différents) ;
  relecture exploratoire aux conditions d'A1 : environ 18/35.
- 31 arrêts du superviseur, tous relancés de zéro et terminés (note 17).
- **G0-S : RECIPE-INSENSITIVE** (règle de la note 13) : T8-S positif échoué (K\*A:F 1,16 ; X:C\* 1,13 ;
  seuils 2,0), nul réussi ; aucune courbe réelle. Voie S fermée ; pas un négatif sur le mécanisme (plafond
  de l'instrument, note 14, écrite avant les données). Option M exclue (exige G0 PASS). G0′ non exécuté
  (note 19). **v84 s'arrête à la faisabilité ; G1b PROCEED en est le résultat positif.**
- Leçons pour v85 : simuler le plafond atteignable d'un instrument (avec un a priori oracle) avant d'en
  figer les seuils ; l'« horloge du savoir » plafonne vers 1,2-1,6× ; le contenu « conséquences » ne se
  transfère pas entre jeux dans la vue égocentrique (exploratoire v83) ; V2 est l'acquis réutilisable.

### 24 octobre (~12 h) — PAUSE demandée par Jérémie ; point de reprise
- **État** : v84 close (faisabilité ; voir l'entrée précédente). Panel de conception v85 terminé, sorties
  dans `docs/panels/v85_*` (brief, 5 propositions, juges, brouillon + audits, `v85_plan_final.md`).
  **Rien n'est figé pour v85** : ni worktree, ni protocole, ni calcul lancé.
- **Plan v85 proposé** (`docs/panels/v85_plan_final.md`) : premier test G1 « le savoir a-t-il de la place
  dans la montée de V2 ? », en ligne (retours en arrière facturés), tous les bras repartant d'une même
  sauvegarde au palier M1 ; trois instruments évaluateurs (choisir parmi ses propositions O_4, où repartir
  K\*map, comment bouger K\*mac) ; étape I de calibration (accélérations plantées ×1 à ×2 dans le régime
  réel) avant de figer les seuils ; puis seulement un objet transportable (familles W ou M), répétition en
  développement, test scellé unique. Chances écrites : place ≈ 35 %, gain transporté confirmé ≈ 1 %.
- **Pour reprendre** : relire le plan final et les audits, décider de l'adopter (ou l'amender), créer le
  worktree `../ragnarok_v85` depuis la branche `arc4-v84-clock`, figer `ARC4_V85_PROTOCOL_EN.md` + version
  française (graines neuves vérifiées), puis coder dans l'ordre de la partie B du plan.
- **Machine** : aucun calcul Ragnarok en cours ; superviseur mémoire arrêté (le relancer caché avant
  tout calcul : `pythonw -u scripts\ram_supervisor.py --low 12 --high 14 --clow 12 --chigh 14`, via WMI
  Win32_Process.Create avec ShowWindow=0, depuis `F:\dev\ragnarok`).

### 3 novembre — v85 : protocole figé (aucun code ni calcul encore)
- Worktree `../ragnarok_v85` (branche `arc4-v85`, depuis `arc4-v84-clock`) ; protocole qui fait foi
  `docs/ARC4_V85_PROTOCOL_EN.md` (= plan final du panel `wf_8db018d6-25c`, SHA-256 `6408f26d…`), version
  française `docs/ARC4_V85_PROTOCOL.md` ; commit `6ef0e05`. Graines 9800-9899 vérifiées neuves.
- **Prochaine étape** : coder dans l'ordre de la partie B du protocole, en commençant par
  `ragnarok/evaluation/senses_v85.py` (départ depuis s1, bac à sable non facturé et non tracé, plafond
  relatif au départ de l'horloge, images oracle), puis `start_v85.py` (S0), `agent_v85/explorer2s.py`
  (V2s), `oracle_v85.py`, `ckpt_v85.py` (points de reprise toutes les ≤ 5 min), `scripts/v85_job.py`,
  `v85_stage0.py`, `v85_calib.py`, `v85_verdict.py`, `queue_v85.py` ; tests de construction 1-9 avant
  l'étape I. Rigueur proportionnée : revue contradictoire avant l'étape I et avant G1 seulement.
- Relancer le superviseur caché avant tout calcul (voir l'entrée du 24 octobre).

### 3 novembre (suite) — v85 partie B : étapes 1 et 2 codées et testées
- `ragnarok/evaluation/senses_v85.py` (commit `d4cd5f8` dans `../ragnarok_v85`) : `start_from` (reset,
  NOOP, restauration de s1, horloge et plafond relatifs), `sandbox()` (non facturé, non tracé ; clone,
  reset et free interdits ; tout restauré), Trace85 (colonne images oracle, `replay85` depuis s1),
  `to_state`/`from_state` selon §13.3. 7 tests verts (breakout, qbert), dont trajectoires identiques
  avec ou sans passages en bac à sable, et reprise identique.
- `ragnarok/evaluation/start_v85.py` (commit `9ff7fe9`) : les 16 unités, ordre `[9801, 1]`, rejeu de la
  lignée du haut (deux contextes), s1, cache du Body, identité du départ, règle des paliers comptés (A.4).
  3 tests verts, dont un vrai départ Alien A.
- `ragnarok/agent_v85/explorer2s.py` (commit `0103fef`) : V2s (A.6) depuis s1, flux par expansion,
  `propose` sans effet de bord, rollout « à sec » pour le bac à sable, crochets injectés (steer, q5,
  chooser, boundary), état de reprise. Tests verts : déterminisme, arrêt à l'expansion 17 puis reprise
  identique à l'octet, propositions et rollouts à sec sans aucun effet.
- `ragnarok/evaluation/oracle_v85.py`, partie 1 (commit `d45c311`) : O_m (coup d'œil gratuit par
  proposition dans le bac à sable, puis la gagnante jouée sur un hasard neuf) et K\*map (carte de la
  lignée, pilotage « où repartir », couverture). Test 6 en partie : O_1 et K\*map à carte vide
  identiques à V2s à l'octet. Correction des tests : l'ancien prélude aléatoire finissait la partie de
  Breakout (rollouts d'un pas) ; désormais vies vérifiées et archive > 20 cellules exigée.
- **Prochaine étape** : finir `oracle_v85.py` (D, O_4-dice, K\*mac et K\*mac-exact, placebos GF4, GF,
  K̃mac ; spécification A.7 et §4.4), puis `ckpt_v85.py`, `scripts/v85_job.py` et la suite de la
  partie B. Aucun calcul lancé ; superviseur toujours arrêté.

### 3 novembre (nuit) — rangement des worktrees (accord de Jérémie) et push GitLab
- Jérémie a autorisé la suppression des dossiers inutilisés et demandé des commits et push réguliers.
- Retirés après commit des petits résultats texte et push de leur branche (gros fichiers pkl/npz/png
  supprimés avec le dossier) : v73 (`arc4-v73`), v74 (`arc4-v74`), v75 (`arc4-v75-offline`), v80
  (`arc4-v80-theory`), v80c (`archive/v80c`), v81 (`arc4-v81-atlas`), v83run (`archive/v83run`, dont les
  JSON de la salle A1), v84s (`arc4-v84-s`). Une branche se rouvre par `git worktree add ../ragnarok_vXX <branche>`.
- Restent : `ragnarok` (master), `ragnarok_v84` (ses données `craft_v6_out/v84` sont lues par v85),
  `ragnarok_v85` (en cours), `ragnarok_v83` (à retirer après le workflow v85 en cours, dont un test v84
  lit encore les données A1). Branches `arc4-v83-promise`, `arc4-v84-clock`, `arc4-v85` poussées.

### 3 novembre (nuit) — v85 partie B construite ; revue contradictoire
- Workflow `wf_00ae4471-8ab` (6 agents, fichiers disjoints) : instruments (D, O_4-dice, GF4, GF, K\*mac,
  K\*mac-exact, K̃mac, Q5 côté agent), points de reprise, registre des bras, `scripts/v85_job.py`,
  `queue_v85.py`, `v85_verdict.py`, `v85_calib.py`, `v85_stage0.py`, tests G-build 1-9 (95 tests verts,
  2 sautés : bras F_M et W de l'item 12, écrits à F1). Commit `9a57b94`, poussé.
- `ragnarok_v83` retiré à son tour (branche `arc4-v83-promise` poussée). Restent `ragnarok`, `ragnarok_v84`
  (données lues par v85) et `ragnarok_v85`.
- Revue contradictoire avant l'étape I : workflow `wf_b15288ac-6a2` (4 relecteurs, 4 vérificateurs,
  1 correcteur). Défauts déjà connus à corriger : lecture de `starts.json` et ordre des unités dans le
  verdict, échecs « dés » non comptés, recherche de graines à étendre à toutes les branches git.
- **Mémoire** : seulement ~9 Go libres (Photoshop ~6 Go) ; ne lancer S0 / l'étape I qu'avec ≥ 14,5 Go
  libres (portes de `queue_v85.py`) et le superviseur relancé caché depuis `F:\dev\ragnarok`.

### 3 novembre (matin) — revue faite et corrigée ; S0 en attente de mémoire
- Revue `wf_b15288ac-6a2` : 34 trouvailles, 27 confirmées, toutes corrigées avec un test qui échoue sans la
  correction (bloquantes : G1-P pouvait être sauté et l'aiguillage ouvrir une famille sans contrôle placebo ;
  le contrôle D comptait comme échecs les unités retirées par l'ajustement croisé). Suite v85 : 124 verts,
  2 sautés (item 12). Commit `6322959` (+ date corrigée), poussé.
- `docs/ARC4_V85_BUILD_NOTES.md` (worktree v85) : choix d'interprétation des passages ouverts du protocole,
  figés avant toute donnée ; à relire par Jérémie. Règle : écrire les items 12-13 sur une branche à part et
  ne fusionner qu'en l'absence de calcul en cours (arbre modifié = Interruption).
- Superviseur relancé caché (`--low 14 --high 15 --clow 14 --chigh 15`, garde v85 de §13.3). S0 lancé par un
  lanceur caché (scratchpad `s0_launcher.py`) qui attend ≥ 14,5 Go libres sur min(RAM, commit) ; journal
  `ragnarok_v85/craft_v6_out/v85/s0/stage0.log`. Si le lanceur a disparu, relancer
  `python scripts/v85_stage0.py run` depuis `F:\dev\ragnarok_v85` (reprend unité par unité).
