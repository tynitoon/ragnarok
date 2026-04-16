# Research Directions — Ragnarok au-delà du workshop

**Rédigé :** 2026-04-16, pendant que le pipeline GPU overnight (paire 3 cleanup + Band B rescue seeds 47-51) tourne.
**Origine :** demande de Jeremie de cartographier les prochaines pistes d'exploration sur trois questions fondamentales du projet. Sortie produite par 4 agents spécialisés (archéologie codebase + Q1 physique + Q2 sélection + Q3 accélération).

## Cadre

Ragnarok est un agent modulaire à RSSM (Dreamer-like) qui **cristallise** des skills (RSSM core + prior + posterior + trunk) et **transfère** un subset vers une nouvelle tâche via `load_state_dict`. Pilot #2 (N=5 seeds, 3 paires) a montré un signal modeste (paire 1 RMST=1.238 fragile, paire 2 négative, paire 3 en re-run). Le workshop paper est à portée *si* Band B ou paire 3 passe ; au-delà, on veut savoir **jusqu'où on peut pousser la méthode**.

Trois questions de recherche portent ce "au-delà" :

- **Q1** — Apprendre un skill dans sa forme *pure*, c'est-à-dire capturer la physique et les interactions causales, pas juste prédire le prochain pixel.
- **Q2** — Choisir le ou les skills à utiliser selon la situation (pas juste un nearest-neighbor figé).
- **Q3** — Vraiment apprendre plus vite grâce aux skills connus, au-delà d'un simple `load_state_dict`.

Ce document est structuré en : (§1) carte de chaleur de la readiness du codebase par question, (§2-§4) pour chaque question — état de l'art, 3 options concrètes, recommandation opinionée, (§5) matrice effort/impact croisée, (§6) roadmap proposée en 3 sprints avec critères d'arrêt, (§7) ce qu'on fait *maintenant* vs ce qu'on diffère.

---

## §1 — Readiness du codebase (archéologie)

| Question | Readiness | Justification en une ligne |
|---|---|---|
| Q1 — Physique pure | **4/10** | RSSM + LatentCuriosity existent, mais aucune factorisation explicite dynamique/bruit, pas d'invariance physique imposée. |
| Q2 — Sélection contextuelle | **5/10** | `SkillSelector` + `CentroidRouter` solides mais statiques ; `LearnedRouter` existe mais n'est *jamais entraîné* (dead code). |
| Q3 — Accélération via skills | **6/10** | `try_transfer()` mature (Bug E, phase 5 warmup), mais init-puis-oublier : pas de kickstarting, pas de reward adaptation, `ewc.py` écrit mais **jamais importé** et API cassée. |

**Signaux positifs** : les dépendances croisées entre les 3 questions sont *toutes positives*. Améliorer Q1 (factorisation z_phys) améliore mécaniquement Q2 (matching plus sémantique) et Q3 (freeze-z_phys durant transfer). Améliorer Q2 améliore l'entrée de Q3. Pas d'architecture bloquante.

**Dette technique détectée par l'audit** :
- `ragnarok/learning/ewc.py` : 79 lignes, jamais importé nulle part, `register_task()` attend un format de loss cassé → **à réparer avant de pouvoir utiliser EWC** (Option Q3-B plus bas).
- `LearnedRouter` (skills/router.py:48-92) : classe définie, jamais entraînée, jamais instanciée → dead code.
- `MultiSkillAgent` : exécution-time seulement, pas de composition learning-time (cf. POST-007).

---

## §2 — Q1 : Apprendre un skill dans sa forme physique pure

### État de l'art (synthèse)

Le diagnostic est solide : la loss de reconstruction d'observation alloue la capacité du world model à la *fidélité* plutôt qu'à la *suffisance causale*. Zhang et al. 2021 (DBC, "Learning Invariant Representations without Reconstruction") montrent que reconstruction s'effondre dès qu'on ajoute des distracteurs ; DreamerV3 (Hafner 2023) admet explicitement que sa stabilité vient de hacks (symlog, twohot) pas d'une structure causale ; Robine 2023 (TWM) documente l'échec OOD sous perturbation physique.

Les sept directions pertinentes :

1. **Object-centric world models** (Slot Attention, SAVi, FOCUS) — décomposer en entités. Gain documenté sur distracteurs, coût VRAM non-trivial.
2. **Contrastive dynamics** (SPR, CURL, DINO-WM, TD-MPC2) — remplacer reconstruction par prédictivité latente. TD-MPC2 est la référence SOTA actuelle.
3. **Neural ODE / Hamiltonian / Lagrangian** (HNN, LNN) — paramétrer la dynamique comme ODE avec conservation d'énergie. Brille sur pendule, casse sur contacts.
4. **Causal representation learning** (Schölkopf, Locatello) — surhype côté RL, théoriquement impossible sans intervention.
5. **Equivariant architectures** (EGNN, MDP Homomorphic Nets) — imposer SO(n) ou E(n). Bon ROI sur 3D, marginal sur nos envs.
6. **Graph / Interaction networks** — overkill pour contrôle low-dim.
7. **Structured state-space models** (S4, Mamba) — gains sur horizons longs, nos épisodes sont courts.

### Trois options pour Ragnarok

**Q1-A — Contrastive RSSM (TD-MPC2 style).** Remplacer la loss `recon_obs` par contrastive latente : prédire `z_{t+k}` depuis `(h_t, a_{t:t+k-1})` via predictor MLP, matcher via InfoNCE/cosine contre target-encoder EMA. Decoder retiré. Reward et continue predictors restent. Transparent pour le transferable subset actuel. **Effort M (1-3 semaines)**. Risque : collapse de représentation si pondération mal calibrée. Falsifiable : sur cartpole→mcc, RMST > baseline à p<0.1 ; et/ou gain >30% sur benchmark distractor synthétique.

**Q1-B — Hamiltonian structured prior (HNN-in-RSSM).** Partitionner z en (q, p), paramétrer dz/dt via Hamiltonien scalaire `H: R^32 → R`. GRU core devient intégrateur d'ODE. **Effort L (1+ mois)**. Risque majeur : nos envs ont des contacts (CartPole tombe, MCC a un mur) qui cassent l'hypothèse Hamiltonienne. Biais inductif devient biais *faux*.

**Q1-C — Contrastive + disagreement-driven ensemble.** Combiner Q1-A avec l'`EnsembleRSSMCore` qui **existe déjà** dans rssm.py:112. Pondérer la loss contrastive par le désaccord inter-cores (apprendre plus fort où les cores divergent). Decoder gardé en régularisation faible (weight 0.1). **Effort S-M (1-2 semaines)** — on réutilise l'infra ensemble. Risque : deux losses en tension (alignement vs reconstruction faible). Falsifiable : (a) RMST delta >10% vs baseline sur cartpole→mcc ; (b) gain >30% sur distractor synthétique — si seulement (a) passe, le gain vient de l'ensemble pas du contrastif.

### Recommandation Q1

**Q1-C**, sans hésiter.

Trois raisons : (1) effort/information imbattable (on réutilise l'ensemble existant, ~300 LoC d'ajout), (2) risk-adjusted supérieur à Q1-A (qui invalide tout le pipeline recon) et à Q1-B (qui casse sur contacts), (3) narrative workshop vendable : "on étend le disagreement ensemble au-delà de l'exploration vers le learning signal lui-même" avec ablation nette on/off et test OOD distractor pour la figure-4.

Si H1 échoue, Q1-C devient un pivot paper "better world models for cross-action-space transfer".

---

## §3 — Q2 : Sélection de skills selon la situation

### État de l'art (synthèse)

Aujourd'hui `SkillSelector.select()` : 10 warmup steps, mean `h_t`, argmin L2 sur centroïdes. Un choix statique, mono-skill, pas d'online re-ranking. La décomposition utile du problème :

- **Temporalité** : task-level (avant épisode) vs state-level (pendant) vs phase-level (tôt vs tard dans training).
- **Cardinalité** : mono (winner-take-all) vs multi-skill (top-k, gated, composition).
- **Provenance** : routeur prescrit (centroïde L2) vs appris (MLP, meta-learned).

Les directions pertinentes :

1. **Options framework** (Sutton-Precup-Singh 1999, Option-Critic Bacon 2017) — chaque skill = (policy, termination, initiation). Gratuit : β_ω donne un critère de "quand re-router".
2. **Feudal hierarchies** (FuN Vezhnevets 2017, HAC Levy 2019) — manager émet sous-but latent, worker exécute. Conceptuellement proche : notre centroïde est déjà un proto-sous-but.
3. **Mixture of Experts gating** (Shazeer 2017, Switch Transformer) — devient intéressant post-10 skills seulement.
4. **PEARL / context-based meta-RL** (Rakelly 2019) — q(z|τ) infère task embedding depuis trajectoire. Version évoluée de notre warmup.
5. **DIAYN / SMERL** — skill discovery via MI. Pas pour nous (skills créés par cristallisation, pas par MI).
6. **Policy-over-policies MCP** (Peng 2019) — mixture multiplicative Gaussienne. Crash sur notre action discret/continu mixte.
7. **Language-conditioned routing** (SayCan, RT-2) — hors scope, pas de langage dans nos envs.
8. **Successor features + GPI** (Barreto 2017-2020) — théoriquement solide mais exige φ partagé cross-env, non-trivial.
9. **Skill priors** (SPiRL Pertsch 2020, OPAL) — ré-exploitable via re-jeu des trajectoires de cristallisation.

### Trois options pour Ragnarok

**Q2-A — PEARL-style context encoder.** Nouvelle classe `ContextRouter(nn.Module)` : prend les 10 (obs, action, reward, next_obs) du warmup, GRU → embedding z_task (dim 64). Compare à un task embedding par skill (nouveau champ dans `Skill.task_embedding`), softmax sur distances. Remplace `SkillSelector.select()`. Entraîné supervisé après chaque cristallisation (identité tâche connue au training time). **Effort S (~3 jours)**. Risque : overfitting à N=3 skills — geler jusqu'à ≥8 skills. Falsifiable : sur 5 nouvelles tâches (library à 10 skills), ContextRouter vs CentroidSelector, **≥20% reduction episodes-to-threshold, p<0.05**.

**Q2-B — Option-Critic avec termination learnée.** `TerminationHead` sur chaque skill chargé : prend h_t, produit β_ω ∈ [0,1]. Dès que β_ω dépasse un seuil (ou sample Bernoulli), re-route via meta-policy. Meta-policy entraîné par policy gradient, β par termination gradient (Bacon eq. 10). **Effort M (~2 semaines)**. Risque : instabilité classique (β dégénère à 0 ou 1) ; mitigation = deliberation cost Harb 2018 (ξ=0.01). Falsifiable : sur tâches *composites* (ex: switch CartPole↔MountainCar mid-épisode), atteinte du plateau en **≤70% des episodes** d'un mono-skill.

**Q2-C — Soft ensemble top-k au niveau exécution.** Dans `MultiSkillAgent.act()`, utiliser `select_soft(h_t)` existant pour obtenir {name: weight}. Action continue → mélange linéaire ; action discrète → mélange au niveau logits puis argmax. Sparse τ=0.1. **Effort S (~2 jours)**. Risque : interférence destructive (skills en désaccord → action moyenne zéro) ; mitigation = top-2 sparse + température basse. Falsifiable : sur 5 tâches x 5 seeds, ensemble top-2 vs mono, **gain ≥10% OU pas de régression >5%**.

### Recommandation Q2

**Q2-C d'abord, puis Q2-A, différer Q2-B.**

Q2-C a un ratio dev/risque imbattable (2 jours, zéro paramètre nouveau, flag-activable). C'est le test empirique pas-cher à faire avant d'investir dans de l'infra — si le gain est nul, on retire en 30 min ; si positif même marginal, ça valide que l'info multi-skill n'est pas perdue au routing.

Q2-A est le vrai pari technique mais demande ≥8 skills pour être statistiquement évaluable, donc attend mécaniquement Post-1 (scale horizontal). À implémenter *pendant* la phase de génération de skills, pas après.

Q2-B (options-framework) en **non-prioritaire** : instabilité β + workshop paper ne le nécessite pas. On y revient seulement si on observe empiriquement des épisodes où le skill initial devient inadapté mid-run.

---

## §4 — Q3 : Apprendre plus vite grâce aux skills connus

*Référence détaillée :* `reviews/transfer_acceleration_review.md` (document séparé, 1479 mots, livré par l'agent Q3).

### Synthèse

Les 4 axes orthogonaux : **quoi** transférer (poids, représentations, policy, value, rollouts, prior exploration, shaping), **comment** (copie, régularisation KL/L2-Fisher, distillation, composition, meta-init), **quand** (init, régularisation continue, guidage décroissant), **métrique** (episodes-to-threshold, AUC, sample efficiency). Notre statu quo occupe une seule cellule (poids, copie brute, init, RMST). On peut ouvrir plusieurs autres cellules sans tout refaire.

Neuf approches couvertes dans le review détaillé : Progressive Networks, EWC/SI, Policy Distillation, MAML/Reptile, PEARL, SPiRL, Successor Features+GPI, Kickstarting, Imagination-based transfer. MAML et PEARL sont hors-scope (besoin >30 tâches meta-train). Progressive Networks tue la scalabilité. Les trois plus prometteuses pour notre setup : Kickstarting, EWC (si on répare ewc.py), Imagination-priming via RSSM transféré.

### Trois options

**Q3-A — Kickstarting par distillation décroissante.** Teacher gelé après `try_transfer()`. À chaque SAC update : `L_kick = λ(t)·KL(π_student ‖ π_teacher)`, λ(t)=λ₀·exp(-t/τ), τ≈20% du budget. **Effort M (3-5 jours)**. Cross-action-type problématique (KL Categorical↔TanhNormal non définie) → action-bridge head nécessaire pour paire 2. Falsifiable : episodes-to-first-solve cartpole→mcc réduit 180→≤150 ; RMST paire 1 passe 1.24→≥1.40.

**Q3-B — EWC branché sur subset transférable.** Prérequis : **réparer l'API cassée de `ewc.py`** (`data_loader_fn` actuel yield des tensors de loss opaques, doit yield (obs, action, next_obs)). Post-crystallisation, calculer F diagonal sur RSSM-core (prior+posterior+GRU), sérialiser dans `Skill.fisher_state_dict`. Au training : `L_ewc = λ·Σ F_i (θ_i - θ*_i)²` au world model loss. **Effort S-M (2-3 jours)**. **Natif cross-action-type** — régularise le RSSM-core qui est cross-env par construction. **Sauve probablement paire 2.** Risque principal : Fisher plat en RL (gradients bruités → F quasi-constant → pénalité inutile). Test cheap : compute F sur skill existant, si max/median <10 EWC ne distinguera rien. Falsifiable : paire 2 passe <1 → ≥1.05 avec N=5 ; variance paire 1 réduite 0.21→≤0.12.

**Q3-C — Imagination-priming via RSSM transféré.** Après `try_transfer()`, **avant la première interaction réelle**, K=5000 rollouts imaginés dans le RSSM transféré sur états initiaux synthétiques, entraîner policy target via SAC sur ces rollouts. Reward predictor fresh ou substitué par RND curiosity (variante Plan2Explore). **Effort M-L (~1 semaine)**. Risque : hallucination distribution shift (rollouts hors manifold target). Falsifiable : episodes-to-first-solve réduit ≥25% sur ≥2/3 paires, N=5.

### Recommandation Q3

**Q3-A en priorité, Q3-B en parallèle si budget permet, Q3-C en plan B.**

Le statu quo est "init puis oublier" — le skill source disparaît après `load_state_dict`. Kickstarting adresse directement ce gap en gardant le teacher actif avec influence décroissante ; c'est le plus proche voisin de l'existant, pas de nouvelle abstraction architecturale. Compatible same-action-type direct (paires 1 + 3). Pour paire 2 (discret→continu) qui nécessite un action-bridge, Q3-B qui opère sur RSSM-core *en aval* de l'action space la sauve indépendamment. Les deux options sont orthogonales et ablatables séparément → double surface de claim du paper.

Q3-C intellectuellement la plus novelle ("cross-env imagination priming" — pas publié explicitement à ma connaissance) mais risque de distribution shift élevé. Plan B si Q3-A+Q3-B sous-performent.

---

## §5 — Matrice effort / impact croisée

Légende effort : S (<1 sem), M (1-3 sem), L (>1 mois).
Légende impact : workshop (aide paper actuel), post-workshop (booster Post-1 scale), fondamental (change nature du projet).

| Option | Effort | Impact workshop | Impact post-workshop | Impact fondamental |
|---|---|---|---|---|
| Q1-A contrastive RSSM | M | Moyen | Fort | Fort |
| Q1-B Hamiltonian | L | Nul | Faible | Fort si marche |
| **Q1-C contrastive+ensemble** | **S-M** | **Fort (ablation propre)** | **Fort** | **Moyen** |
| Q2-A PEARL context encoder | S | Faible (N=3) | Fort (N≥8) | Moyen |
| Q2-B Options-critic | M | Nul | Moyen | Faible |
| **Q2-C soft ensemble top-k** | **S** | **Moyen (cheap win)** | **Moyen** | **Faible** |
| **Q3-A Kickstarting** | **M** | **Fort (peut sauver paire 1)** | **Fort** | **Moyen** |
| **Q3-B EWC branché** | **S-M** | **Fort (peut sauver paire 2)** | **Fort** | **Moyen** |
| Q3-C Imagination-priming | M-L | Moyen (risque) | Fort | Fort si marche |

**Gagnants dominants** : Q1-C, Q2-C, Q3-A, Q3-B.

---

## §6 — Roadmap proposée

Dépend du résultat overnight. Trois branches comme dans la décision workshop :

### Branche A — Band B ≥ 1.30 OU paire 3 clean 5/5 (résultat positif modeste)

1. **Sprint 1 (2 semaines)** — Q3-A Kickstarting sur paire 1 (N=5 seeds, same action space). **Kill si RMST < 1.15 sur 3/5 seeds.** Si ≥1.40, écrire le paper B0 en y intégrant Kickstarting comme contribution méthodologique secondaire.
2. **Sprint 2 (2 semaines, en parallèle)** — Q3-B EWC branché (réparation ewc.py incluse). Cible paire 2 (cross-action). **Kill si Fisher max/median < 10** (test cheap en Day 1).
3. **Sprint 3 (post-paper)** — Q1-C contrastive+ensemble, nouveau pilot 3-paires avec N=5. Benchmark OOD distractor comme claim secondaire.
4. **Post-paper** — Q2-A PEARL encoder pendant Post-1 scale (nouveaux skills qui le nécessitent).

### Branche B — Band B ∈ [1.20, 1.30) p<0.10 + A11 pass (modeste)

Identique à branche A mais avec **Q3-A en avant du paper** (pas en secondaire). Si Kickstarting fait passer paire 1 de 1.24 à 1.40+, on peut re-raconter le paper comme "skill transfer + kickstarting distillation = robust cross-action transfer". Narratif plus fort que B0 pur.

### Branche C — Rien ne passe (pilot effondre)

Pas de paper workshop immédiat (aligné avec ton intuition). Pivot :
1. **Sprint 1 (2 semaines)** — Q3-B EWC seul (cheap, S-M). Si Fisher plat, skip.
2. **Sprint 2 (2 semaines)** — Q1-C contrastive+ensemble. Re-run paires avec nouveau world model.
3. **Sprint 3+ (Post-1 horizontal scale, 5-10 nouvelles tâches)** avec Q2-A au fur et à mesure.
4. Paper main-track 3-6 mois plus tard, beaucoup plus fort que B1 pur-négatif.

### Critères d'arrêt communs (toutes branches)

- Tout sprint n'est commité que si pré-spec métrique + kill criterion écrits avant de lancer (prereg amendment).
- Budget GPU par sprint : ≤20 GPU-heures (comparable à pilot #2 = 12.65h).
- Multi-agent review à chaque gate (cf. calibration : "dissent > consent").

---

## §7 — Ce qu'on fait maintenant (pendant overnight)

**Aucun code**. Le pipeline GPU tourne ; on ne touche pas à la branche master. Ce qu'on peut faire sans code :

1. **Lire & rebondir** sur ce document + sur `reviews/transfer_acceleration_review.md`. Identifier les choix qui te hérissent, les approches que tu veux challenger.
2. **Pré-enregistrer** les métriques falsifiables de Q3-A et Q3-B comme amendement §13 v3.7 — ça nous engage sur les seuils avant de voir les résultats. Cohérent avec la rigueur preregistration actuelle.
3. **Spawner une review devil's-advocate** sur ce document : est-ce que je surpitche Q1-C ? Est-ce que Q3-B passe vraiment le test Fisher ? Les métriques sont-elles assez strictes ?

**Ce qu'on diffère au matin** (résultats overnight) :
- Décision branche A / B / C.
- Si A ou B : lancement Sprint 1 (Q3-A Kickstarting).
- Si C : décision paper vs pivot direct Post-1.

**Ce qu'on ne fait pas tant qu'on n'a pas pris la décision branche** :
- Toucher au code Ragnarok.
- Lancer un autre pilot.
- Écrire le paper.

---

## §8 — Références pour approfondir

**Q1** — Zhang 2021 DBC (arXiv 2006.10742), Hafner 2023 DreamerV3 (2301.04104), Hansen 2024 TD-MPC2 (2310.16828), Schwarzer 2021 SPR (2007.05929), Zhou 2024 DINO-WM (2411.04983), Greydanus 2019 HNN (1906.01563).

**Q2** — Bacon-Harb-Precup 2017 Option-Critic, Vezhnevets 2017 FuN, Rakelly 2019 PEARL, Barreto 2017 Successor Features, Peng 2019 MCP, Shazeer 2017 sparse MoE.

**Q3** — Rusu 2016 ProgNets, Kirkpatrick 2017 EWC, Czarnecki 2019 Distilling, Schmitt 2018 Kickstarting, Pertsch 2020 SPiRL, Barreto 2020 GPI, Sekar 2020 Plan2Explore.

---

*Document vivant. Prochaine update : au matin, après lecture des résultats overnight et décision branche A/B/C.*
