# Ragnarok — point de reprise

Mis à jour le 9 septembre 2026 vers 19:54 Paris. Continuer d'un jalon au suivant,
sans terminer à chaque commit. Arrêt uniquement demandé par Jérémie, travail
achevé, ou décision exigeant son avis. L'audit partage d'écran/RC a été annulé.
Objectif : apprendre puis réutiliser pour apprendre plus vite. Une avance initiale
ne prouve pas une accélération. ARC 2 clos : transfert 3/6 contre 0/6 ; accélération
non détectable (delta 0,069, bruit 0,076). ARC 4 : aucun entraînement Atari ni gate 0.

Historique technique détaillé : `docs/history/CODEX_HANDOFF_20260909_1935.md`, puis
celui de 18:11 et de 16:48. Lire uniquement les sections utiles ; pas tous les logs.
Les captures de calibration sont réservées à l'instrumentation, jamais à l'agent.

## ARC 3 v62 — confirmation GPU active

Vérifié 19:27 : contrôleur 20936 `C:/Python314/python.exe -u -m
scripts.recover_confirm_v62`, enfant 22896 `scripts.confirm_v61 --lineage 1
--n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume`.
État `craft_v6_out/v62_confirm_recovery_state.json` : running/confirm_s1.
Lignée 0 complète (3 mondes/9), 25433 s ; agrégat et 14 checkpoints validés contre
le contrôleur, puis committés 9deb244. SHA agrégat 3d30c06f….
Lignée 1 démarrée 18:41. À 19:53, log s1_8310 : M4 trois objectifs terminés à
2101 s ; Fa deuxième objectif terminé à 3648 s. Aucun verdict intermédiaire.
Vérifier les commandes ET `craft_v6_out/v61_confirm_s1.log`, puis s2.

Reprise à 11:37 Paris après Windows Update ; bras incomplet rejoué mêmes graines.
Interruption facturée 2166 s (ancien 9366 erroné préservé/corrigé). Budget de reprise
84742,0902837 s. Projection initiale 34,3461 GPU-h/42, pas une garantie.
Preuves/recovery b815be2, `v62_confirm_interruption_20260909/`.
Autorité : `ARC3_PLAN.md` §12.6 ; budget/preuves §§13.1–13.5. B6/N9/3 lignées/G'2,
store persistant pour tous les bras. Gel a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
Aucun changement des 17 sources scientifiques, des 3 lanceurs, états/checkpoints.
Aucun deuxième entraînement GPU. Gate/probe PASS revus, pas verdict confirmatoire.
Après `confirm_complete_review_required` : sources, tous les mondes, checkpoints,
budget, revue indépendante, score gelé, puis vraie figure. Figure corrigée/revue
423b770/af94a62 sur données synthétiques seulement. Suivi horaire existant
`suivi-ragnarok-arc-3` : ne pas dupliquer ni créer de nouvelle tâche.

## ARC 4 — instruments achevés, difficultés conservées

Six calibrations brutes et temporelles PASS/revues ; sources scellées immuables.
Références CPU : Pong ≈1356, Breakout ≈1005, Freeway ≈1431, Seaquest ≈1312 décisions/s.
MsPacman complet : NumPy 552,129/s, natif 580,793/s, 171364 charges chacun ;
16 rapports/digests strictement égaux, 58 constructions, 2042 images de branche.
Sous 1000 et 2000 proposés : problème d'ingénierie à résoudre avant GPU ARC 4.
Revue finale `v70_mspacman_cpu/cpu_chain_result_review_v0.json`, SHA40f640d4…,
115 sources/233 preuves, commit ebf4a80. Ces 115 sources/DLL/build ne bougent pas.
Pas de débit universel : mesures séquentielles sous charge ARC 3. Tous les processus
CPU des benchmarks/captures sont terminés ; aucune collecte native actuellement lancée.

Ast CPU v0 FAILED : Unknown player graphics bank au décodage anticipé RAM frame0,
265 charges, aucune décision, aucun retry/substitut/benchmark v0. Échec cbc0ef2.
Prototype séparé `deferred_visible_source_v70.py`, revue source 01313bf8….
Audit offline complet sur les 8192 images déjà archivées (17571) : 2049 groupes,
scènes/membres/historique identiques ; 225 groupes contiennent plusieurs actions,
donc ce n'est pas un rollout natif repeat4. Revue finale b43a95f5… à ebf4a80,
121 sources/38 preuves. Aucune nouvelle émulation Ast autorisée par ce seul audit.

Traces RGB natives PASS : 9 cas, 26 instances, 32 constructions, 22604 charges.
Sources 20 gelées 87c0507 ; préflight f0d8a22a… committé avant lancement ; résultats,
tentative, console exacte et revue 64e82d1b… à 0e23739. Seul Seaquest a une différence
non nulle ; validité du transport, pas contrôlabilité apprise. Ne pas modifier
ces 20 sources, y compris `docs/ARC4_BRANCH_TRACE_DRAFT.md` (texte historique).
Archive privée + transport : nouveaux prototypes relus, pas validés nativement.
`rgb_archive_v70.py` : paire sur archive 2T+180, retour au point courant ; pas de
retour durable E4. Capacité totale pool 20000 entrées/2 Go de payload, pas de RSS.
Revues préliminaires source afbe5846… (11 sources/15 tests), transport 36731edf…
(16 sources/23 tests) ; sources 7a7fe35/817bb97/eaad4c0 inchangées.

## ARC 4 — apprenant en construction, uniquement CPU synthétique

Nouveaux modules dans `ragnarok/learning/`, sans enseignant RAM/palette/référence.
- Parser + P + instances : 428677 paramètres, 49 floats/instance, déplacement
  prédit par intervalle, 12 effets fixes, zéro=effet inconnu. Mapping boutons→effets
  non résolu. Revue préliminaire f6d18381… (10 sources/17 tests), sources d728bae.
  Limites conservées : P peut manquer l'intérieur uniforme et marquer un fond
  nouvellement couvert ; groupes k-means possiblement disjoints.
- Tracker : revue 5d25929f… (9 sources/11 tests), source b9e1a1d. Centres pixel
  exacts, extrapolation selon horizon explicite, mémoire4/coast30, rayonnement8px,
  affectation maximale puis coût minimal, coasts distincts du visible, update atomique.
- Adaptateur borné : 512 pistes coasts compris, refus atomique sans éviction.
  C : projette les pixels propres dans la différence future, branches incomplètes
  inconnues ; support bilinéaire partiel renormalisé, différence À l'horizon.
  Revue commune `counterfactual_capacity_review_v0.json` SHA1d9dc215…,
  19 sources/14 tests, source e7d228a/doc599fcec, reçu committé d6521d9.
- G/I : grouping_v70.py / identity_v70.py, doc ARC4_EMBEDDING_OBJECTIVES_DRAFT.
  G utilise m/d propres détachés, voisins4/tolérance0,5pixel, variance par groupe
  et marge1. I : InfoNCE visible à lag8 exact, même flux/époque, négatifs obligatoires.
  Revue PASS `embedding_objectives_review_v0.json` SHAcb41aa1e…(11 sources/17 tests),
  reçu committé93e23c8. Défaut reshape géométrique corrigé dansd6521d9 ; domaine
  float32/64 seulement à37e5d90 (gradientfloat16 pouvaitdéborder, AMPnonvalidé).
  Collapse exact G et I stationnaire, contre-exemples gardés.
  Ancres admissibles ne prouvent ni gradient ni entraînement.
- Assemblage perception_stream_v70.py initial879b624, finaldd08dbd/doccd0fc78,
  8 tests PASS, revue indépendante EN COURS parreview_recovery_v62.
  Images synthétiques, max des deux dernières RGB, G/I + transition P séparée.
  Horizon fixé4, P inconnu si transition1..3, I exact8. Mémoire pixels uint8/trackers
  sans graphes ; réévaluation avec poids courants ; erreurs tardives atomiques.
  C reste séparé, pas de collecteur attestant actions/délais/provenance.
  Corrections93e23c8 : zéroP tronqué sûr (pas sumoverflow), gardepertesfinies
  avantadmissibilité. dd08dbd : tagsstream/epochreadonly etorigineconservée dans
  chaqueentrée ; nouveaumémoiresurresetagentdécidé, jamaisresetALEoracle.
  Fenêtre[t−8,t]inclusive : jusqu'à9observations, pas8échantillons.
- Catégorieslocales `visual_categories_v70.py` +test/doc à34036e9 :5testsPASS,
  NONencorerelu.42features[f32,logsize2,RGB8],distance.6,max64catégories,
  refusatomiqueaucunefusionforcée,coastsexclus. Exportlocalexcludel'héritage.
  Fusioncomportementale nonimplémentée,aucunevalidationAtari.

Le pool existant rend du RGB poolé ordinaire, pas directement les deux dernières
images brutes ni une chronologie de collecte. Les métadonnées d'époque doivent
suivre les réinitialisations de mémoire décidées par l'agent, pas devenir un
signal de mort oracle issu de RAM/ALE. Résoudre cette interface avant collecte.
Aucun ME/buttons/reader/drives/S/policy/ledger apprenant implémenté.

## Diagnostic CPU externe en cours

ProfilageofflineMs17471 terminé, session46226 fermée. Sources/protocole46f42d1
avant exécution ; résultats/tentativecommittésb56dfa0 dansv70_mspacman_cpu/
offline_scoring_profile_v0.json.8192images,2051groupessynthétiques,329multiactions.
Composants avecsurcoûtprofileur : decode13,148s,grouping7,319s,histoire4,439s,
matching3,139s,référence1,531s ; loop30,949s.0émulation,aucundébitnatifdéduit.

Nouveauxadaptateursdujugeàb56dfa0 : vector_visible_mspacman_v70.py (imageintégrale,
ancienclippingocclusion),cached_visible_groups_v70.py (dernièreentréecomplètepar
type,membrescompris).5testsPASS :256masques4×2,100populationsrectangles,groupes
ordonnés/cache/identités/validation. Les115sourcesoriginellesintactes.
AuditexhaustifOFFLINE ENCOURS, session30037 : scripts.check_visible_acceleration_v70.
Sources/protocole860319dcommittésAVANTlancement ;40965pairesDEV/RES/raw17471,
24576groupes,3collections8192. Sortieprévuev70_visible_cpu/vector_cached_equivalence_v0.json
et_attempt. Aucunautreprofilage/natif/GPU. Ne modifier les sources/preuves auditées.
Aprèsfin : reçus/sources/digests/équivalenceindépendante, puis seulement protocole
prospectif de nouvellechaînenative. Ne pas annoncer580→nouveaudébitsansmesure.

Suite utile : terminer revue assemblage, revuecatégories, équivalenceCPUsecteur ;
intégration C/collecteur sansoracle. Préserver les
sources relues, créer des adaptations prospectives, aucune relance native improvisée.
`ARC4_VISION.md` §§2,5,6,11 et `ARC4_ENTRY_BUDGET_DRAFT.md` : 29 paires E1 de8images
sur16 points courants, 3944 charges expérimentales /19720 totales, pas16×18.
Protocole de collecte/gate encore à figer ; mort ALE Pong/Freeway sans vies : régler
l'éligibilité avant gel. Gate0 proposé≤12 GPU-h, aucun lancement avant résolution,
revue et gel. Boxing/Frostbite/Riverraid/Kangaroo jamais ouverts.

## Environnement et préservation

Master, commits ciblés, pas de push ni suppression. Ne pas lire `.codex/config.toml`.
Exclure `.codex/`, sorties ARC3 actives (`v61_confirm.json`, s1/s2/états), anciens smokes.
`SETUP.md` : C:/Python314/python.exe Torch2.11.0+cu126 ; aussi utilisé explicitement
sur CPU pour tests Torch, CUDA_VISIBLE_DEVICES vide, OMP/OPENBLAS/MKL=1, torch threads1.
.venv-arc4/Scripts/python.exe : Python3.14.3/ALE0.12.1/numpy2.4.4/scipy1.18.1,
pytest9.0.3/pillow12.3.0, sans Torch. Ryzen53600X 6c/12t, RTX4080 16Go.
Réutiliser review_recovery_v62 aux jalons ; audit_gate_v62 ancien annulé.
