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
