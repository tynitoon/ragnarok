# Plan v71 classé : défauts confirmés et changements génériques

Synthèse des six axes et de leurs contre-vérifications. Je n'ai lancé aucun processus python. J'ai vérifié les noms de fonctions dans `ragnarok/agent_v71/learner.py` et `policy.py`.

## 0. Constat global (mesuré)

- Aucun défaut corrigé n'a produit, seul, un gain de score live qui sorte du bruit.
- Les meilleures politiques apprises atteignent au mieux le meilleur bouton constant, à l'intérieur du bruit (Assault BRI-hold32 : 374 et 351 contre UP 403 et 397 sur 10 parties). Elles ne le dépassent pas.
- Le bruit de mesure est du même ordre que tous les écarts entre bras :
  - même graine, même code, Demon Attack : 128, 235, 455 (non-déterminisme GPU) ;
  - erreur type d'une moyenne sur 5 parties : environ 40 à 70 sur Assault, 150 à 200 sur Carnival.
- La référence « UP 476 » sur Assault vient de 3 parties seulement. Sur 30 parties, elle vaut environ 414 ± 21.
- Sur Carnival, la référence « meilleur bouton constant = 0 » est faible : alterner deux boutons au hasard donne 568, l'aléatoire uniforme 496.
- Fait utile : les événements de score lus dans le HUD collent au vrai score (Assault : 166 vrais positifs, 5 faux positifs, 0 manqué). L'agent peut donc mesurer seul, sans récompense ni RAM, que tenir un bouton bat sa politique.

## 1. Défauts réels (vérification passée), classés par impact mesuré

| # | Défaut | Où | Impact mesuré | Statut |
|---|---|---|---|---|
| D1 | Le fit est sans masque, et la tête de sortie est transportée d'un jeu à l'autre. Un jeu de pré-entraînement où seul « none » existe (Name This Game, dernier jeu dans 3 checkpoints sur 4) fait saturer la sortie : p(none)=1.000, entropie ≈ 0. | `policy.py Policy.fit` ; `learner.py Agent.learn` (appels à `fit`) | Le transfert à l'entrée s'effondre : Assault 13 contre 229 pour l'agent neuf (3 graines : 16.8, 8.4, 16.8). Test causal : 300 pas de fit sans masque suffisent à faire s'effondrer un réseau neuf ou déjà entraîné ; avec masque, rien ne bouge. | Confirmé, cause du transfert négatif à l'entrée |
| D2 | La table d'effets à l'entrée est mal construite. Les 12 sondes tombent dans les pas 0 à 72 après un reset, des boutons différents sont fusionnés, et `to_button` tire au hasard parmi les boutons fusionnés. | `learner.py Agent.enter` (boucle `range(12)`) ; `me_buttons.py effect_table` ; `policy.py EffectAdapter.to_button` | Name This Game : 0 bouton utilisable, donc la politique n'a aucun effet (deux politiques non entraînées donnent exactement 1894, comme tous les bras pilotes). Assault : UP est fusionné dans -x avec LEFT et LEFTFIRE, et jouer -x en constant donne 176 contre environ 414 pour UP : plafond structurel. Score live après correction : inchangé dans le bruit (DA 163→166, NTG 1894→2227, Assault 338→290). | Confirmé ; préalable, pas le goulot |
| D3 | La démo est rejouée depuis un nouveau `env.reset()`, pas depuis la sauvegarde racine de l'explorateur. | `learner.py Agent.learn` (ligne `obs = env.reset()`) ; `Agent.explore` ne garde pas la racine | Assault : la démo vue par `learn()` vaut 315 au lieu de 462 ; restauration exacte depuis la racine. Live 298 et 340 → 374 et 361 : environ 1 à 2 erreurs types, dans le bruit. Seul Assault diverge sur les 6 jeux. | Confirmé |
| D4 | Les entrées de clonage comportemental (BC) sont construites avec `perc.locate` (sondes et réparation du fond). Les rollouts et le jeu live utilisent `tracker.update` ou `reacquire`. Les sondes faites pendant le rejeu modifient aussi le résultat. | `learner.py Agent.learn` (boucle sur `demo`) | Entrées différentes sur 57.8 % des pas sur Demon Attack et 2.9 % sur Assault. Rejeu exact rétabli (DA 60→295, Assault 210→273), mais ce test est presque tautologique. Live DA, 3 graines : 143→557 (une partie à 1900), 161→177, 295→252. | Confirmé comme défaut, gain non établi |
| D5 | L'auto-imitation ne change pas le choix des actions : le mélange d'actions reste figé sur celui de la démo, et l'entropie vaut 0.17 à 0.33 nats. La sélection de segments repondère surtout les états, pas les actions. | `learner.py Agent.learn` (boucle `rounds`, médiane des segments) | Assault, base : mélange r00 [.23,.20,.54,0,.03] puis r19 [.21,.22,.51,0,.06] ; live 252→265. AWR, rollrank et eps 0.3 : aucun gain fiable. | Confirmé ; c'est le vrai goulot de la compétence |
| D6 | Le tracker perd l'identité de « moi » sur Breakout : il saute sur la balle à t=136, et `track_miss` masque l'erreur. | `tracker.py MeTracker.update` (pénalité de taille souple) | Erreur d'identité de 100 % après la 1re mort (sticky 0). Avec le « moi » oracle, aucun gain (3.8 et 4.2, puis 2.0 et 2.4, contre 3.2 à 4.6). | Réel, non limitant |
| D7 | Les étiquettes d'effet ne sont pas canoniques entre jeux, ni même entre graines d'un même jeu. | `me_buttons.py effect_table` | Assault s0 : RIGHTFIRE = -y ; s1 et s2 : spawn. FIRE vaut -y sur DA et -x sur Carnival. Cela ruine le sens de la tête transportée. | Confirmé |

**Réfuté ou non soutenu :**
- H4 (effondrement sur un seul bouton) : c'est l'inverse, le mélange d'actions reste figé.
- H5 (écart entraînement/live) : la politique est déjà sous le constant dans ses propres conditions d'entraînement (Freeway 84 %, paire significative). Sur Assault, l'écart par vie est nul.
- « Sur Demon Attack, la politique apprise est pire que l'aléatoire » : non soutenu, le même bras a donné 455.
- Explorateur « hold » : aide Demon Attack mais nuit à Breakout et Carnival.
- sticky + reset dans l'auto-imitation, et le filtre de taille du tracker : négatifs sur Breakout.

## 2. Changements génériques, dans l'ordre

Chaque étape doit passer son contrôle avant la suivante. Le critère de réussite est fixé avant de lancer.

### Étape 0 — Protocole de mesure (scripts d'expérience, pas l'agent)
- **Quoi :**
  - recalculer les références avec au moins 10 parties × 2 graines live (7 et 8), sticky 0.25, un env neuf par action ;
  - références : chaque bouton constant, l'aléatoire uniforme, et l'alternance aléatoire de deux boutons (pour Carnival) ;
  - pour les bras appris : au moins 2 reprises d'entraînement par graine, car le GPU n'est pas déterministe. Couper cuDNN rend le calcul 35× plus lent : faire des reprises plutôt que forcer le déterminisme ;
  - mettre en cache entrée et exploration (`selfimit/cache.py`) pour que les bras partagent la même démo.
- **Preuve que c'est nécessaire :** 128, 235 et 455 à graine et code identiques ; UP 476 sur 3 parties contre 414 ± 21 sur 30.
- **Critère :** un bras ne « bat » une référence que s'il la dépasse de plus de 2 erreurs types en moyenne poolée.

### Étape 1 — Fit masqué (D1)
- **Où :** `policy.py Policy.fit(imgs, poss, acts, steps, bs, mask=None)`.
- **Quoi :** appliquer `masked_fill(~mask, -1e9)` aux logits avant `cross_entropy`. Dans `learner.py Agent.learn`, passer `mask=self.adapter.available` aux deux appels à `fit`. Ne rien entraîner si une seule sortie est disponible.
- **Preuve :** `verify_transfer/collapse_test.json` (masqué : distribution inchangée) ; MPolicy, stade Carnival (EMa) : entropie 0.16 ; transfert au budget 0 = 208.
- **Contrôle :** relancer le pré-entraînement leave-one-out puis l'entrée sur Assault, graines 0, 1 et 2, live 7, 10 parties. Attendu : E ≈ F (environ 200 à 240) au lieu de 13. Contrôle statique : `ckpt_stats.py`, entropie masquée supérieure à 0.7·ln(k).

### Étape 2 — Table d'effets fidèle (D2, D7)
- **Où :** `learner.py Agent.enter` ; `me_buttons.py effect_table` ; `policy.py EffectAdapter`.
- **Quoi (version C testée, `diag/effects/adapters.py SignatureAdapter`) :**
  - (1) 24 sondes prises toutes les 100 étapes de jeu ordinaire, plus seulement dans les pas 0 à 72 ;
  - (2) garder le signe du déplacement, diagonales comprises ;
  - (3) un bouton dont la signature de différence d'image (horizons 2 et 4) diffère du bouton le plus simple de sa classe reçoit son propre emplacement ;
  - (4) `to_button` devient déterministe : un emplacement correspond à un bouton, sans tirage au hasard parmi des boutons fusionnés.
  
  L'alternative D (sondes réparties, 6 effets canoniques) est codée dans `run_one.py` mais n'a pas été lancée.
- **Preuve :** 6 boutons sur 6 distincts sur DA, NTG, Carnival et Breakout (2 graines, reproduit). Assault : UP a son propre emplacement.
- **Contrôle :**
  - (a) NTG : deux politiques non entraînées (torch 11 et 12) doivent donner des scores live différents. C'est un test de validité, pas de compétence ;
  - (b) sur les 8 jeux, compter les boutons adressables par graine (0 et 1) ;
  - (c) Assault : jouer constamment l'emplacement de UP doit égaler UP (environ 400 sur 10 parties).
  
  Coût : environ 43 000 frames de plus par jeu.
- **Limite connue :** sur Chopper Command (défilement), la table reste bruitée. Le signe du déplacement n'est pas fiable : NTG RIGHT mesuré à dx=-7.75. Ne pas prétendre que les étiquettes sont canoniques entre jeux.

### Étape 3 — Rejeu de la démo depuis la racine (D3)
- **Où :** `learner.py Agent.explore` doit renvoyer aussi `root = env.clone()` pris juste après `env.reset()`. `Agent.learn(..., root)` fait `env.restore(root)` au lieu de `env.reset()`. Adapter `run_game` en conséquence.
- **Preuve :** Assault 315 → 462, restauration exacte, deux fois.
- **Contrôle (déterministe et peu coûteux) :** sur les 8 jeux, graine 0, vrai score de la démo rejouée = vrai score de la démo à l'exploration. Côté expérimentateur seulement.

### Étape 4 — Rejeu de la démo avec la perception du jeu live (D4)
- **Où :** `learner.py Agent.learn`, boucle `for i, a in enumerate(demo)`.
- **Quoi :** un seul `perc.locate` pour initialiser, puis `perc.tracker.update(obs) or perc.tracker.reacquire(obs)`. Aucune sonde ni réparation de `perc.bg` pendant le rejeu ; geler `perc.bg` après l'entrée.
- **Preuve :** écart d'entrées 0.578 → 0 ; rejeu en boucle fermée DA 295, Assault 273 (= démo).
- **Contrôle :** DA, Assault et Breakout, 3 graines d'entraînement × 10 parties live (graines 7 et 8). Critère : pas de régression ; aucun gain n'est attendu (passé : +414, +16, −43). Ne pas utiliser la fidélité du rejeu comme preuve de compétence.

### Étape 5 — Auto-imitation par branches contrefactuelles (D5 : le vrai goulot)
- **Où :** `learner.py Agent.learn` (remplacer la sélection de segments par la médiane), actions au niveau bouton grâce à l'étape 2.
- **Quoi (BRI-hold32, `diag/selfimit/bri.py`) :**
  - depuis des états visités par la politique, brancher chaque bouton, le tenir 32 pas sans continuation ;
  - noter chaque branche avec les événements de score du HUD de l'agent, moins une pénalité de vie, et imiter la meilleure ;
  - ajouter au pool les segments « bouton tenu » quand ils marquent plus vite que la politique.
- **Preuve :** Assault, part de UP .30 → .83 ; final 407.4 sur 5 parties, 374 et 351 sur 10 parties, contre UP 403 et 397. C'est la seule règle qui déplace le mélange d'actions. Carnival : 17 à 23 racines sur 24 sont à égalité, donc pas de signal.
- **Ce qui manque :** un mécanisme pour dépasser le constant (horizons courts suivis de la politique, valeur par un critique appris). Non testé.
- **Contrôle :**
  - jeux : Assault, Freeway, Demon Attack, Name This Game (après l'étape 2), Carnival ;
  - 2 graines d'entraînement × 10 parties × graines live 7 et 8 ;
  - références : meilleur bouton constant poolé, aléatoire uniforme, aléatoire à deux boutons, politique non entraînée ;
  - critère fixé d'avance : dépasser de plus de 2 erreurs types le meilleur constant sur au moins 2 jeux. Sinon, résultat négatif à garder tel quel.

### Étape 6 — Transfert (seulement si l'étape 5 donne une compétence)
- **Où :** pré-entraînement et entrée (`diag/transfer/soft.py`, `xfer.py MPolicy`).
- **Quoi :** fit masqué (étape 1) ; ne pas transporter la tête quand les tables d'effets ne concordent pas (sinon encodeur seul, Ec) ; plancher d'entropie à l'entrée comme filet de sécurité (ET).
- **Preuve :** au budget 0, Assault ET 231 et Ec 256 contre F 229 (inoffensif). Au budget 300, F bat ET sur 3 graines appariées (+105, +29, +34) : aucun gain d'apprentissage.
- **Contrôle :** leave-one-out sur 4 unités, budgets 0, 100, 300 et 1000, 3 graines. Mesurer la vitesse d'apprentissage, pas seulement l'avance à l'entrée.

### Étape 7 — Tracker (D6, basse priorité)
- **Où :** `tracker.py MeTracker.update` et `reacquire`.
- **Quoi :** filtre de taille strict, plus un test de contrôlabilité : « moi » est perdu s'il ne bouge pas comme la table d'effets le prédit. Journaliser le taux d'erreur d'identité, pas `track_miss`.
- **Contrôle :** erreur d'identité de Breakout après la 1re mort sous 0.2, sans régression de score. Aucun gain de score attendu.

## 3. Points spécifiques à un jeu ou proches d'un oracle

- Aucun correctif testé n'utilise la RAM, le vrai score ou le nom du jeu dans sa logique. Les contre-vérifications l'ont confirmé pour SignatureAdapter, la racine, le rejeu par tracker, BRI, ET/Ec et le fit masqué.
- `scorer.py player_coord(game, ram)` est spécifique au jeu et lit la RAM : réservé à l'évaluation, à ne jamais importer dans l'agent.
- Le « moi » oracle (référence par sondes injectée dans la politique, axe livegap) a servi au diagnostic seulement.
- `env.clone()` met `_true_score` dans le tuple de sauvegarde (`s[3]`). C'est acceptable tant que l'agent ne le lit jamais ; à couvrir par un test d'assertion.
- `EffectAdapter` suppose NOOP = action 0 : convention ALE, pas une connaissance d'un jeu, mais à signaler.
- « Tenir un bouton 32 pas » est un horizon fixe générique, pas réglé par jeu. Ne pas le régler jeu par jeu après observation.

## 4. Règles mémoire et processus pour l'implémentation

- **Un seul processus python à la fois** pour cette session, sans pool, sans file d'attente, sans relance automatique.
- **Contrôle de mémoire dans la commande elle-même**, plus seulement affiché : lancer seulement si plus de 14 GB sont libres, revérifier chaque minute jusqu'à 15 minutes. Utiliser par exemple `diag/effects/gate.sh`. Trois oublis passés : lancements à 13.7, 12.17 et 11.9 GB parce que la valeur était affichée sans bloquer.
- Garder au moins 12 GB libres pour la session Tavern en permanence. Le superviseur RAM a déjà suspendu une exécution environ 16 minutes : prévoir de la marge dans les durées.
- Chaque expérience doit durer moins de 20 minutes : run_game au budget 300 ≈ 550 à 590 s ; « continuer à travers les morts » a dépassé ce plafond (arrêté à 1250 s). Découper entrée/exploration (en cache) et apprentissage/évaluation en lancements séparés. Stockage en uint8.
- Ordre de coût estimé : étapes 1 à 4 ≈ 8 à 12 lancements courts ; étape 5 ≈ 5 jeux × 2 graines × environ 15 minutes, en série.
- Avant de rendre la main, lister les python.exe dont la ligne de commande pointe vers le dossier scratch et n'arrêter que ceux-là. Ne jamais toucher F:/dev/tavern.

Tous les scripts et résultats cités sont dans `C:/Users/jeremie/AppData/Local/Temp/claude/F--dev-ragnarok/826ba484-cdf9-442a-89b7-2049c18871fc/scratchpad/diag/` (sous-dossiers `effects/`, `demo/`, `fidelity/`, `selfimit/`, `livegap/`, `transfer/` et les `verify_*` correspondants).