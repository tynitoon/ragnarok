# Ragnarok — point de reprise court

Mis à jour le 9 septembre 2026 vers 16:48 Paris. Index, pas nouveau protocole.
Détails précédents conservés dans [l’historique de 16:48](history/CODEX_HANDOFF_20260909_1648.md)
et celui de 14:30. Lire uniquement les sections utiles, pas la session Claude entière.

## Instruction active

Jérémie a levé la pause et demande de poursuivre d’un jalon au suivant sans
terminer la tâche. S’arrêter seulement au terme du travail, sur sa demande,
ou pour une décision qui nécessite réellement son avis. Les commits servent
à continuer. L’audit partage d’écran/RC a été annulé comme erreur de session.

But : montrer qu’un agent apprend, conserve ses acquis et apprend ensuite plus
vite. Une avance initiale ne démontre pas une accélération. ARC 4 vise une
perception apprise, puis agence, commandes, motivations, mémoire et transfert :
Atari → NES → 16 bits → PC. Aucun apprenant ARC 4 ni gate 0 n’a été lancé.
RAM, palettes et référence classique servent à l’évaluation externe ; jamais
aux observations ni à l’enseignement du futur agent.

## ARC 3 v62 — calcul confirmatoire actif

Contrôle réel à 16:47 : contrôleur **20936**, enfant **17340**, GPU 31 %, 4238 MiB.
Commandes : `C:/Python314/python.exe -u -m scripts.recover_confirm_v62`, puis
`scripts.confirm_v61 --lineage 0 --n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume`.
État `craft_v6_out/v62_confirm_recovery_state.json` = running/confirm_s0.
Vérifier les commandes et logs, pas seulement les PID ou un ancien état running.
`s0_8300` terminé à 8326 s ; `s0_8301` entièrement terminé après G’ à 18089 s.
`s0_8302` commencé, objectifs [3,1,0], tiers [2,3,4], nav min 0,883.
Deux mondes terminés sur neuf ; aucun verdict avant les trois lignées et revue.
Logs réels `v61_confirm_s0.log`, puis s1/s2 ; stderr contrôleur vide au contrôle.

Reprise démarrée le 9 septembre à 11:37:39 Paris après interruption Windows
Update à 06:30 et boot à 06:39. Bras incomplet rejoué avec les mêmes graines,
sans sélection ; **2166 s** facturées (ancien 9366 erroné préservé/corrigé).
Budget de reprise **84742,0902837 s** ; délai strict du contrôleur inchangé.
Projection initiale totale 34,3461 h sous 42 GPU-h, pas une garantie de durée.
Preuves `v62_confirm_interruption_20260909/`, revue/recovery, commit b815be2.

`ARC3_PLAN.md` §12.6 fait autorité ; §§13.1–13.5 donnent preuves et budget.
Tag gel `v62-stage0-freeze` = a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
Design B6/N9/3 lignées/G’2, store persistant entre objectifs pour tous les bras.
Ne modifier aucun des **17 fichiers scientifiques gelés**, des trois lanceurs
`resume_v62`, `confirm_resume_v62`, `recover_confirm_v62`, checkpoints ou états.
Ne pas lancer un second entraînement. Les coupes ne se redécident pas après lancement.

Gate et probe PASS déjà revus. Probe M4 = 618/768, R = 398/768, delta 0,286458
> 3SE 0,072446. Trois préentraînements : 22823,7297333 s ; probe : 1087,029983 s.
Douze AUC reproduites. M3 n=123 : ni accumulation monotone ni loi parfaite établie.
Après confirmation : sources/checkpoints/budget, revue, score gelé puis figure.
`figure_v61.py` non gelé corrigé, commit **423b770** : labels poids + store aux
objectifs suivants, n du probe visibles, grille sans chevauchement. Rendus
**synthétiques uniquement** dans `v62_figure_layout_check_v0/v1/v2` ; v0/v1
négatifs visuels préservés, v2 inspecté PASS. Les 17 sources gelées sont intactes.
Suivi horaire existant `suivi-ragnarok-arc-3`, silencieux hors changement utile ;
ne pas dupliquer, mettre en pause à la clôture ou si avis utilisateur requis.

## ARC 4 — les six instruments bruts et temporels sont validés/revus

| Jeu | Bruts et repeat4 réservés | Référence / débit CPU |
|---|---|---|
| Pong | PASS, revus | terminé ; rappel brut ≈99,69 %, natif ≈1356 décisions/s |
| Breakout | PASS, revus | terminé ; ≈99,20 %, ≈1005 décisions/s |
| Freeway | PASS, revus | terminé ; ≈99,74 %, ≈1431 décisions/s |
| Seaquest | PASS, revus | terminé ; ≈93,64 %, ≈1312 décisions/s |
| Ms. Pac-Man | PASS, revus | sources prêtes, préflight final en cours |
| Asteroids | PASS, revus | sources prêtes, préflight final en cours |

Mesures d’instruments, pas de perception apprise. Débits sous charge ARC 3,
comparaisons séquentielles ; pas de gain universel ni de gate de débit déclaré.
Détails, anciennes versions négatives et écarts : `docs/ARC4_<JEU>_CALIBRATION.md`,
protocoles POOLED associés, `ARC4_SEAQUEST_REFERENCE.md`, `ARC4_NATIVE_REFERENCE.md`.
Moteur froid, charges, snapshots natifs/RGB/sticky et replays documentés dans
`ARC4_COLD_START.md` et `ARC4_RESET_AUDIT.md`. Sources scellées immuables.

Dernières preuves : Ast17522, 8192 décisions / 34596 charges, trois catégories
PASS ; revue **15b8931c…**, données committées a14b07d. Ms17422 V2, 8192 décisions,
32753 frames + six démarrages ×132 = **33545 charges**, cinq catégories PASS.
Sources Ms **02b5dc7**, scellé **9ebdfdf** avant capture, données **e7334f7**,
revue finale/fingerprints **c77b860**. Revue Ms :
`e8cfc5c2b2ec041edfd795b427ed12b8fcaf3db913094b3047a9624d60d048cf` ;
résultat : `8e6f930e5aae2e5cc96fff193fe046f03e01d1ce6168941c2192930cc0df8da4`.
57 sources / 87 preuves / 64 archives ; 30 témoins reproduits, dont les gommes
avec historique RGB causal. Résidus Ms : 26 ghost/1 pill en comptage,
60 ghost/16 fruit/5 pill en géométrie. Ils restent présents, aucune nouvelle correction.
Toutes les sessions de capture/score antérieures sont terminées.

## Suite immédiate : références fraîches, puis CPU

Relecteur `/root/review_recovery_v62` actif sur le préflight **sans ALE** de
`scripts/visible_reference_v70.py`, protocole `ARC4_VISIBLE_REFERENCE_PROTOCOL.json`.
Les deux fingerprints sont remplis dans `visible_temporal_prerequisite_v70.py`.
99 sources committées ; quatre vraies preuves raw/temporal PASS et 34 tests PASS.
Reçu attendu `craft_v6_out/v70_visible_reference/preflight_review_v0.json` :
passed, source_sha256 exact, runtime, raw_proof_sha256 et temporal_proof_sha256
avec les deux jeux. Vérifier/committer avant les vrais check/capture.

Graines **17471 Ms / 17571 Ast toujours fermées**. Ensuite, une seule capture
par jeu avec `python -m scripts.visible_reference_v70 --game <jeu> --stage capture`,
puis `--stage score` ; faire `--stage check` avant. 8192 frames brutes, actions
uniformes tenues 16 frames, vrai précédent RAM y compris frame 0. Sorties
`v70_<jeu>_calibration/raw_reference_v0*`, marqueurs capture/score distincts.
Référence RGB causale continue entre épisodes ; RAM seulement pour le scorer.
Juge source corrigé MsV2/AstV1 sans nouvelle validation brute indépendante :
les métriques restent conditionnelles ; ne pas transformer le PASS temporel en nouveau raw PASS.

CPU préparé : `bench_visible_reference_v70.py`, `check_visible_cpu_v70.py`,
`visible_reference_pool_v70.py`, histoire RAM restaurable et groupes raster.
Commit **2f0cd99**, 112 sources prévues, 45 tests PASS ; revue préliminaire
**bd117616…** (à actualiser après empreinte Ms), aucune ouverture de graines.
`REFERENCE_REVIEW_SHA256=None` bloque tout jusqu’à revue finale des deux références.
Barrières : preuves des deux jeux, sources/build/DLL committés et exacts,
rapports déjà reçus conservés sur panne, arrêt définitif après erreur,
empreintes géométrie/wiring/paire inchangées après travail, digest ordonné des
RGB/boîtes ordinaires et de branche. Coût constructeur sans retour explicitement inconnu.

Après références revues/committées : remplir hash, revue CPU finale et commit,
équivalence exacte sur toutes les 8192 fenêtres développement + 8192 réservées
par jeu, commit ; wiring **18176/18177/18178** payé, commit ; NumPy puis natif
**16×2048 décisions**, graines **18000+i**, même trajectoire et rapports/digests
exacts. Garder toute équivalence négative. Aucun essai réservé relancé/substitué.
Frame 0 fournit la géométrie, pas l’éligibilité au mouvement ; pilules statiques.

## Après les instruments

Implémenter parser/tracker/ME/buttons/reader/drives selon `ARC4_VISION.md` §§2,5,6,11.
Descripteur corrigé à 49 floats (addition), pas encore implémenté. Aucun gate/poids ARC 4.
`ARC4_ENTRY_BUDGET_DRAFT.md` : 29 paires E1 au maximum dans 20k à20 %, pas 288 ;
16 points courants, 3944 charges expérimentales / 19720 totales en simulation.
Archive persistante, traces intermédiaires, choix des points et partage avec C restent ouverts.
Avant gel : registre graines/jeux, allocation E1/couverture, événements de mort
éligibles (Pong/Freeway sans vies ALE, pas de faux F1 parfait), plafond proposé
12 GPU-h. **Boxing/Frostbite/Riverraid/Kangaroo restent fermés**.
Aucun entraînement ni état d’entrée d’agent extrait des captures d’instrumentation.

## Environnement et préservation

Branche master ; commits ciblés, pas de push. Ne rien supprimer ni lire les
configurations locales, notamment `.codex/config.toml`. Exclure les sorties
ARC 3 actives, anciens smoke tests et fichiers locaux non suivis des commits.
`SETUP.md` actualisé : GPU `C:/Python314/python.exe`, Torch2.11.0+cu126 inchangé ;
CPU `.venv-arc4/Scripts/python.exe`, Python3.14.3, ale-py0.12.1, numpy2.4.4,
scipy1.18.1, pytest9.0.3, pillow12.3.0, sans Torch. Bench OMP/OPENBLAS/MKL=1.
Ryzen5 3600X (6c/12t), RTX4080. Ne pas réinstaller ou reconstruire un binaire scellé.
Réutiliser le relecteur existant aux jalons ; ancien `audit_gate_v62` annulé après reboot.
ARC 2 clos : transfert 3/6 vs0/6 ; accélération non détectable (delta0,069/bruit0,076).
