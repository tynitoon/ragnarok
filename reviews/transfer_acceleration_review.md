# Accélérer l'apprentissage via skills cristallisés — au-delà de `load_state_dict`

**Contexte :** pilot #2, N=5 seeds. Paire 1 cartpole→mcc RMST=1.238
(fragile, seed-46-driven), paire 2 acrobot→cartpole-swingup RMST<1
(négative), paire 3 pendulum→cartpole-swingup en re-run. Transfert
Ragnarok actuel = init statique + training standard, sans régularisation
ni guidage online. Ce doc challenge cette architecture et propose 3
options testables au-delà de la composition multi-skill déjà listée
en POST-007.

---

## 1. Cadrage

"Apprendre plus vite via skills" se décompose sur quatre axes orthogonaux :

- **Quoi transférer :** poids (statu quo), représentations gelées,
  policy (distillation), value function (successor features),
  rollouts pré-computed (seed buffer), prior d'action π_skill(a|s),
  shaping de reward, prior d'exploration.
- **Comment :** copie brute, régularisation (KL, L2 Fisher-pondéré),
  distillation off-policy, composition (MoE, lateral connections),
  meta-init (MAML / Reptile).
- **Quand :** à l'init (statu quo), pendant tout le training (régularisation
  continue), guidage online décroissant (kickstarting, λ(t) → 0).
- **Métrique cible :** episodes-to-threshold (ce qu'on mesure via RMST),
  sample efficiency à budget fixe, AUC normalisée. Zero-shot n'est
  pas le claim ici — on veut *learn faster*.

Le transfert actuel occupe une seule cellule (poids, copie brute, à
l'init, RMST). Les options ci-dessous en ouvrent d'autres.

---

## 2. État de l'art

**Progressive Networks** (Rusu 2016). Chaque task ajoute une colonne
avec lateral connections vers colonnes précédentes gelées. Zéro
forgetting par construction. Démontré Atari/manipulation, positif
sur 5/7 paires. Tue la scalabilité : params = O(n) pour n skills —
à 10 skills, 10× le baseline scratch.

**EWC / Synaptic Intelligence** (Kirkpatrick 2017 ; Zenke 2017).
Fisher diagonal F_i estimé post-task ; pénalité Σ F_i (θ_i - θ*_i)².
Marche sur permuted MNIST et Atari sequential ; médiocre quand F est
plat (bruit de gradient RL). **État Ragnarok : `learning/ewc.py`
existe (79 lignes) mais n'est importé nulle part** (vérifié grep
global). Plus grave, son API est cassée : `register_task()` attend
un `data_loader_fn` qui yield *des tenseurs de loss* pré-calculés,
pas des batches — non-utilisable tel quel sur RSSM.

**Policy Distillation** (Rusu 2016 ; Czarnecki 2019). KL(student ‖
teacher) comme loss auxiliaire. Czarnecki montre qu'un mauvais teacher
*ralentit* le student ; exige softmax-temperature ou expected-KL
matching. En SAC off-policy, on distille sur les mêmes batches que
le critic.

**MAML / Reptile** (Finn 2017 ; Nichol 2018). Meta-init θ* qui, après
k SGD steps sur task i, minimise la loss task-i. Exige distribution
de tâches au meta-training (typiquement ≥30). Avec 3 skills,
non-applicable. Hors scope workshop.

**PEARL** (Rakelly 2019). Context encoder q(z|τ) infère task embedding
depuis transitions ; policy conditionnée sur z. Excellent sur
meta-RL (MT10, ML45). Même limite distribution que MAML.

**SPIRL / Skill-Prior RL** (Pertsch 2020). Pré-entraîne p(a_{t:t+H}|s)
sur data offline ; nouvelle task → KL(π ‖ p) penalty. Le prior
n'impose pas *quoi* faire mais *parmi quelles actions* choisir.
Gain 2-5× sur kitchen/maze. **Conceptuellement la plus proche de
Ragnarok** : un skill cristallisé *est* un prior.

**Successor Features + GPI** (Barreto 2017, 2020). r = φ(s)·w ;
apprend ψ^π(s) = E[Σ γ^k φ] ; GPI prend max sur policies source.
Zero-shot quand nouveau reward ∈ span(w_sources). Exige φ partagé —
non-trivial cross-env (acrobot φ ≠ cartpole φ sans apprentissage joint).

**Kickstarting** (Schmitt 2018, DeepMind). λ(t)·KL(student ‖ teacher)
auxiliaire avec λ(t) → 0 décroissant. Le student *dépasse* le teacher
sur DMLab-30. Budget typique 100M-1B frames ; fonctionne en budget
réduit si teacher proche du target. **Techniquement la plus pertinente
pour Ragnarok** : init + guidage online décroissant = exactement ce
qui manque au statu quo.

**Imagination-based transfer** (Dreamer v2/v3 ; Plan2Explore Sekar
2020). L'agent rollout dans le world model, entraîne policy sur
imagined trajectories. Si le world model transfère, la policy apprend
sans coûter d'interactions réelles. Ragnarok a **déjà** `dreamer.py`
+ `dream_augmenter.py` — on pourrait *dreamer* avec le RSSM transféré
avant la première interaction réelle. Pas publié explicitement pour
cross-env à ma connaissance : petit claim possible.

---

## 3. Options concrètes pour Ragnarok

### Option A — Kickstarting par distillation décroissante

- **Inspiration :** Schmitt 2018 + Czarnecki 2019.
- **Mécanisme :** Le skill source reste en mémoire comme teacher gelé
  après `try_transfer()`. À chaque SAC update :
  `L_kick = λ(t) · KL(π_student(·|s) ‖ π_teacher(·|s))` sur les mêmes
  batches que le critic update. Schedule λ(t) = λ₀·exp(-t/τ), τ ≈ 20%
  du budget. Classes touchées : `RagnarokAgent` (stocker
  `self._teacher_policy`), `SAC.update()` (ajouter terme au policy loss).
- **Cross-action-type (discret→continu) :** **Problématique.** KL entre
  Categorical et TanhNormal n'est pas définie. Solution : action-bridge
  head (projection linear + moment-matching), ou restreindre aux paires
  same-action-type. Paire 2 (acrobot discret → cart-swingup continu)
  tombe dans le bridge case.
- **Mono vs multi-skill :** mono natif ; multi = somme KL pondérées par
  distance latente (pas de mode-cancellation puisqu'on moyenne des
  *distributions*, pas des poids).
- **Effort :** **M.** 3-5 jours : teacher storage, hook SAC, schedule,
  tests ; +1-2 jours si action-bridge.
- **Risque :** Mauvais teacher piège le student (Czarnecki documente).
  Mitigation : λ₀ petit (0.1), schedule court, tracking reward
  validation.
- **Métrique falsifiable :** "episodes-to-first-solve cartpole→mcc
  réduit 180→≤150 (N=5), RMST paire 1 passe 1.24→≥1.40." Kill :
  RMST < 1.15 sur 3/5 seeds.

### Option B — EWC branché sur le subset transférable

- **Inspiration :** Kirkpatrick 2017, adapté au subset RSSM que
  Ragnarok identifie déjà (`rssm.load_transferable_state_dict`).
- **Mécanisme :** Post-crystallisation, calculer F diagonal sur
  RSSM-core uniquement (prior + posterior + GRU), sérialiser dans
  `Skill.fisher_state_dict`. Au transfer : charger F avec θ*. Pendant
  le training : `L_ewc = λ·Σ F_i (θ_i - θ*_i)²` au world model loss.
  **Prérequis : réparer l'API cassée de `ewc.py`** — `data_loader_fn`
  doit yield `(obs, action, next_obs)` et reconstruire la loss en
  interne, pas des tensors opaques.
- **Cross-action-type :** **Oui, natif.** Régularise le RSSM-core
  qui est cross-env par construction (Bug E v2) ; indépendant de
  l'action space. **Sauve probablement paire 2.**
- **Mono vs multi-skill :** mono natif ; multi = Σ λ_k·EWC_k normalisé.
- **Effort :** **S-M.** 2-3 jours : réparer `ewc.py`, hook au crystallisation
  (`skills/library.py`), hook au transfer (`agent.try_transfer`), terme
  dans `WorldModelTrainer.update()`.
- **Risque principal :** **Fisher plat en RL.** Gradients RL bruités
  → F quasi-constant → pénalité inutile. Test rapide : computer F sur
  skill existant, si max/median < 10, EWC ne distinguera rien.
- **Métrique falsifiable :** "paire 2 (actuellement <1) passe à ≥1.05
  avec N=5 ; et/ou variance σ paire 1 réduite 0.21→≤0.12." Kill : EWC
  dégrade baseline sur mêmes seeds.

### Option C — Imagination-priming via RSSM transféré

- **Inspiration :** Dreamer (Hafner 2019-2023) + Plan2Explore (Sekar
  2020). Claim nouveau : utiliser l'imagination *à travers le world
  model transféré* pour pré-former la policy target **avant la
  première interaction réelle**.
- **Mécanisme :** Après `try_transfer()` charge RSSM-core + trunk,
  **avant d'envoyer l'agent dans le target env**, K=5000 rollouts
  imaginés avec le RSSM sur états initiaux synthétiques (z samplé du
  prior), entraîner la policy target via SAC sur ces rollouts.
  Reward predictor fresh — on peut soit co-fine-tune, soit substituer
  par RND curiosity reward (variante Plan2Explore). Classes touchées :
  nouveau `scripts/imagination_prime.py` ; hook dans `agent.__init__`
  après `try_transfer()`.
- **Cross-action-type :** **Oui.** Action space = target ; RSSM transféré
  fournit les dynamiques latentes, policy target garde le bon action
  space.
- **Mono vs multi-skill :** mono natif ; multi = imagination avec mixture
  de RSSM-cores (cohérent avec option E de POST-007).
- **Effort :** **M-L.** ~1 semaine : chemin d'imagination, reward
  predictor fresh, validation non-divergence.
- **Risque principal :** **Hallucination distribution shift.** Si le
  RSSM transféré rollout hors manifold du target, policy apprend dans
  le vide. Reward predictor random amplifie. Mitigation : K modéré,
  mixer avec données réelles dès disponibles.
- **Métrique falsifiable :** "episodes-to-first-solve réduit ≥25% sur
  ≥2/3 paires, N=5. Spécifique : paire 1 passe 180→≤135 ep." Kill :
  priming cause divergence (reward < random sur 1ers 50 ep) sur ≥2/5.

---

## 4. Recommandation

**Prioriser Option A (Kickstarting), puis Option B (EWC branché) en
parallèle si budget permet.**

Le statu quo Ragnarok est "init puis oublier" — le skill source est
consommé une fois et disparaît. Kickstarting adresse directement ce
gap en gardant le teacher actif avec influence décroissante. C'est la
plus proche voisine de l'existant (pas de nouvelle abstraction
architecturale) et elle est directement compatible same-action-type.
Pour le workshop : Option A sur paires 1 et 3 (both continuous) est
un win direct ; si ratio 1.24→1.40+, la story devient vendable.
Paire 2 reste hors scope Kickstarting pur (action mismatch) — mais
Option B, qui opère sur RSSM-core *en aval* de l'action space, peut
la sauver indépendamment. Les deux options sont orthogonales et
ablatables séparément, ce qui double la surface de claim du paper.

Option C (imagination-priming) est intellectually la plus novelle,
mais risque élevé de distribution shift ; Plan B si A+B sous-performent.
Le claim paper qui va avec ("cross-env imagination priming") est plus
fort mais plus fragile.

**Premier sprint concret :** 5 jours, Option A sur paire 1
(cartpole→mcc, same action type), N=5 seeds, compare RMST vs pilot #2
baseline. Si ≥1.40, commit pour paire 3 ; si <1.15, pivot vers
Option B.
