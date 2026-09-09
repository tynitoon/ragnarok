# Ragnarok — point de reprise court

Mis à jour le9septembre2026 vers16:02Paris. Index, pas nouveau protocole.
Historique antérieur intégral conservé dans
[docs/history/CODEX_HANDOFF_20260909_1430.md](history/CODEX_HANDOFF_20260909_1430.md).
Lire les sources utiles par sections ; ne pas recharger toute la session Claude.

## Instruction active et cap

Jérémie a levé la pause le8septembre et demande explicitement de poursuivre
sans terminer à chaque jalon. S'arrêter seulement au terme du travail, sur sa
 demande, ou pour une décision qui nécessite réellement son avis. Commits et
points de reprise servent à continuer. La demande d'audit partage d'écran/RC a
été annulée comme erreur de session.

But : montrer qu'un agent apprend, conserve et réutilise ses acquis pour
apprendre plus vite. Une avance initiale ne démontre pas cette accélération.
ARC4 vise une perception apprise par ses propres expériences, puis agence,
commandes, motivations, mémoire et transfert : Atari→NES→16bits→PC.
C'est une ambition ; aucun apprenant ARC4 ni gate0 n'a encore été lancé.
RAM, palettes et perception de référence sont des instruments externes,
jamais des observations ni des enseignants gratuits du futur agent.

## ARC3 v62 — calcul confirmatoire actif

Contrôle réel vers16:01 : contrôleur20936, commande
`C:\Python314\python.exe -u -m scripts.recover_confirm_v62` ; enfant17340,
`scripts.confirm_v61 --lineage 0 --n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume`.
État `craft_v6_out/v62_confirm_recovery_state.json` = running/confirm_s0.
Vérifier commandes et logs réels, jamais PID seuls ni anciens états running.
`s0_8300` et `s0_8301` terminés ; dernier bras Fb objectif18 à15360s cumulées,
puis L=[1,1,1]. Processus actif, GPU46%,5715MiB au contrôle ; troisième unité
en préparation/calcul, son en-tête pas encore écrit. Pas d'erreur contrôleur.
Aucun verdict avant les trois lignées, neuf mondes et revue indépendante.

Lanceur démarré le9septembre à11:37:39Paris après reprise d'une interruption
WindowsUpdate à06:30. Deux objectifs sans checkpoint avaient été interrompus :
bras incomplet rejoué avec les mêmes graines, sans sélection.2166s facturées
jusqu'au dernier boot ; ancien chiffre9366 erroné conservé et corrigé.
Preuves `craft_v6_out/v62_confirm_interruption_20260909/`,
`v62_confirm_recovery_review.json`, commitb815be2 ;8tests de reprise passent.
Budget de reprise84742,0902837s ; projection initiale totale34,3461h sous42GPU-h.
Le délai strict du contrôleur reste applicable même si les calculs ralentissent.

Sources : `ARC3_PLAN.md` §12.6 fait autorité sur les anciennes notes, §13 et fin
`preregistration.md`. Tag `v62-stage0-freeze` = a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
Store persistant entre objectifs du même monde pour toutes les conditions.
Design B6/N9/3lignées/G'2. Ne modifier aucun des17fichiers scientifiques gelés,
aucun des3lanceurs de reprise, états historiques, checkpoints ou preuves actifs.
Ne jamais lancer un deuxième entraînement. Les sorties d'enfants sont dans
`v61_confirm_s0.log` puis s1/s2, pas seulement le stdout du contrôleur.

Gate PASS et probe PASS déjà revus : probe M4=618/768, R=398/768,
delta0,286458>3SE0,072446. Trois préentraînements achevés22823,7297333s ;
probe1087,029983s. Les12AUC ont été reproduites depuis les checkpoints.
M3 n=123 seulement ; aucune accumulation monotone ni loi parfaitement apprise.
À la fin : contrôler sources/checkpoints/budget, revue scientifique, puis score
et figure gelés. Corriger les légendes non gelées qui parlent encore de
weights-only/per-goal store avant livraison. Respecter STOP sans relance cachée.

Suivi horaire existant `suivi-ragnarok-arc-3`, silencieux hors changement utile.
Ne pas dupliquer. Mettre en pause à la clôture ARC3 ou si avis utilisateur requis.

## ARC4 — instruments CPU, aucun apprentissage

Moteur froid certifié sur six jeux, comptabilité complète, sauvegardes/reprises
et sticky RNG testés. Voir `docs/ARC4_COLD_START.md`, `ARC4_RESET_AUDIT.md` et
`ARC4_NATIVE_REFERENCE.md`. Ancienne sous-facturation conservée comme erratum.
Toutes les sources scellées restent immuables ; créer de nouveaux adaptateurs.

| Jeu | Images brutes | Fenêtres repeat4 | Référence et débit CPU |
|---|---|---|---|
| Pong | réservé validé/revu | validé/revu | terminé,≈99,69%rappel brut |
| Breakout | réservé validé/revu | validé/revu | terminé,≈99,20%rappel brut |
| Freeway | réservé validé/revu | validé/revu | terminé,≈99,74%rappel brut |
| Seaquest | paire réservée validée/revue | validé/revu | terminé,≈93,64%rappel brut |
| Ms.Pac-Man | paire17412/17432validée/revue |17421V2PASSrevu, préflight17422clos | à faire |
| Asteroids |17512validé/revu |17522PASSrevu etcommitté | préparé, fermé |

Les rappels de référence mesurent un instrument RGB classique, pas l'agent.
Le débit natif de Seaquest≈1312frames/s reste sous l'objectif proposé2000 ;
Pong≈1356, Breakout≈1005, Freeway≈1431, sous charge ARC3 variable. Rapports
NumPy/natif identiques ; aucun gain universel ou gate de débit déclaré.
Détails et négatifs : `docs/ARC4_<JEU>_CALIBRATION.md`, fichiers POOLED associés,
`docs/ARC4_SEAQUEST_REFERENCE.md` et `ARC4_NATIVE_REFERENCE.md`.

Ms.Pac-Man : raw sources5a2ee619, scellé1fddd2c,31sources+7métadonnées.
Uniforme17412 : fruit strictement absent reste null/FAIL ; quatre types passent.
Couverture17432 : les cinq passent, sans agréger les dénominateurs.
Paire SHA8c0aaedd3f4d70d0ecac4b9681dcdbd5ac91164854c4f7013701b0de3c16359a,
revue a53c16554af0bc91a57ea3876aabb2a26e2323e78793b96b10e29a91fe99ef89,
preuves complètes committéesf13dae6.17411/17431développement et tous négatifs gardés.

Suite Ms.Pac-Man : `docs/ARC4_MSPACMAN_POOLED_PROTOCOL.json` ;17421/17422,
8192décisions repeat4, trois RAM adjacentes pour deux RGB,43tests CPU PASS.
Capture/scorer distincts avec tentatives durables et coût partiel sur échec.
Revue indépendante close,45sources committées1dedc9f avant17421 ; vrai préflight
PASS. Capture complète8192décisions/33416facturées/5loads/99,43s.
V0 complet : quatre types PASS, ghost comptage7956/8068=.986118 FAIL,
géométrie27999/28065=.997648 PASS. Revue d4faf514… confirme preuves et négatif.
V1séparée `pooled_mspacman_rows_v70.py` retire une ligne fantôme àymod12=1,
vide dans1557/1557cas ; RGB et critères inchangés.5tests PASS, recalcul intégral
achevé avec `calibrate_mspacman_pooled_rows_v70.py`, sortie geometry_v1 distincte.
V1ghostFAIL7963/8068comptages positifs,28015/28073within1 ; quatre autres types
identiquesV0. Inventaire complet reproduit tous les stats et105erreurs restantes.
Ce sont les yeux de fantômes mangés, pas du score à filtrer. HypothèseRAMpure
V2état4/6+positionexacte+pointeury+47, glyphe8×5 et occlusion player/fruit :
221/221supports exacts sur deux retours ; témoin état5 rejeté. Nouveau module
`pooled_mspacman_eyes_v70.py`, scorer `_eyes`,10tests PASS ; revue candidat
adcb683e… et52sources committés4660467 avant score. V2 complet PASS : comptes
ghost8192/8192et8068/8068positifs,28159/28176within1 ; quatreautresstatsV1inchangés.
Revue résultat6a6dce50… confirme64shards,52sources,38témoins. RésultatV2SHA
6510be585e16341ef108d315c1493a55ac2a483698b652d2c69004725afec7d3.
Runner réservé `validate_mspacman_pooled_v70.py` maintenantV2,57sources ; revue
préflight41010456… close,16métadonnées,40tests PASS. Résultat/revues/sources àcommitter, puis créer/committer
`pooled_reserved_seal_v2.json`, vrai check, capture unique et score distinct.
Le premier nom CLI V0 entrait en conflit avec le marqueur de capture : refus
avant scoring, données intactes, puis nom geometry distinct.17422fermé.

Asteroids : raw sources73fdf4c, scellé158a976,31sources+8métadonnées.
17512 :8192images/8852facturées, trois types PASS ;23images de comptage rochers
erroné conservées. Résultat SHAd24fe0badfa72fb42601cb7e0e878251ad41db589344bfecc5b32e817408faf9,
revue d019a8992d59ef9b050ebe0bcf489d2783da3ddd39485d4de1465d825b827dc9,
preuves committées0306672. Le premier PASS hors ROI et V1–V4 négatifs sont conservés.
Protocole17521/17522 et candidat/capture/scorer temporels préparés ;17tests
spécifiques PASS. Revue close, SHA973aafd44d8f76cf10f835eb84fd157da0bf4718f35c5ad8ea641a3b2b3c2776.
41sources committées313b8d3, préflightPASS,17521capturé8192/34476facturées/13loads.
V0scoring incomplet sur garde pointer/direction ; marqueur failed conservé.
Trois sources5320/5321/5322 ont des pointeurs d'explosion décalés au wrapy=-2.
V1séparée `pooled_asteroids_wrap_v70.py` les reconnaît sur copieRAM, sans changer
RGB/ROI/seuils ;5tests PASS. Recalcul intégral V1PASS sur8192fenêtres ; revue
indépendante edfdff3b… confirme44sources et dix témoins.31/27/6erreurs de comptage
conservées. Sourcea2fccaf avant score ; reçu commun V1 créé pendant le calcul,
écart documenté sans prétendre à une revue pré-run. Réservé préparé49sources+
9métadonnées, préflight indépendant5bd30d86… et30tests PASS. Sourcecommitc724311,
scellé créé13:30:10Z et committé avant toute construction17522 ; check conforme.
Sources helper `temporal_validation_seal_v70.py` et runner
`validate_asteroids_pooled_v70.py` désormais immuables. Capture/score marqueurs
distincts, pas de relance réservée. Réserve17522 complète :8192décisions,
32748ordinaires+14×132=34596charges, trois catégories PASS. Revue15b8931c…
confirme49sources/64shards/RNG/actions/33témoins et sept wraps nouveauxy=-1/-4.
Erreurs comptes18/24/2 et géométrie1/54/0conservées. Preuves committéesa14b07d.
SealSHAcdd9b4677fd45b51916413d4928113850412abf0e7abff2a6fb5dc21442ed5b9.

Préparation référence non gelée : `docs/ARC4_VISIBLE_REFERENCE_PROTOCOL.json`,
`scripts/visible_reference_v70.py`, `visible_temporal_prerequisite_v70.py`,
`raw_history_fixtures_v70.py`, protocole référence, tests : audit préalable clos.
Scène source utilise MsV2/AstV1 corrigés ; capture nouvelle archiveRAMprécédente
exacte àframe0→1, pas de fallback. Juge composé sans nouvelle validation raw
indépendante : limite inscrite au protocole/résultat. Tous shards réservés relus
au préflight. Empreinte revueAst remplie ; Ms=None bloque LESDEUX jeux jusqu'aux
deux revues finales. Graines17471/17571fermées. Sourceclosure99 et tests ciblés
environ34PASS ; finaliser deux fingerprints, revue globale/commit avant captures.

Préparation CPU committée112a5cb/4d7c806 : `restorable_ram_history_v70.py`,
`visible_temporal_groups_v70.py`, `visible_reference_pool_v70.py`, scènes/tests.
25tests synthétiques PASS, aucun émulateur pour cette chaîne. Snapshot complet
ajoute troisRAM ; branche restitue historique exact. Décodage aprèsframepayée
préserve coût même sur erreur. Groupes raster4CC et membres pour mobilité ;
référence et ciblesC invariantes au retraitoracle dans les tests. Revue, preuve
géométrie complète, wiring réel/protocole/runnerbenchmark encore àfaire.
`ARC4_ENTRY_BUDGET_DRAFT.md` : calcul seul,29pairesE1max dans20kà20%, pas288 ;
16points courants possibles avec13lots×2+3lots×1. Archive persistante/trajectoires
intermédiaires/partageC restent ouverts, aucun gel ni table gratuite revendiqués.

Après les deux calibrations temporelles : référence puis débit, puis primitives
parser/tracker/ME/buttons/reader/drives. `ARC4_VISION.md` §§2,5,6,11décrit la vision
et les blocages avant gel : registre graines/jeux, entrée E1 et couverture,
allocation expérimentale commune, événements de mort éligibles, plafond12GPU-h
proposé. Pong/Freeway sans vies ALE : pas de F1 parfait sur absence de positifs.
Boxing/Frostbite/Riverraid/Kangaroo restent fermés. Aucun gate ni entraînement
sur les captures d'instrumentation qui l'interdisent.

## Environnement et préservation

Branche master ; commits ciblés, aucun push requis. Ne rien supprimer.
GPU `C:\Python314\python.exe`,Torch2.11.0+cu126 ; ne pas modifier.
CPU `.venv-arc4/Scripts/python.exe`,Python3.14.3,ale-py0.12.1,numpy2.4.4,
scipy1.18.1,pytest9.0.3,pillow12.3.0,sansTorch. Voir SETUP.md.
Benchmarks OMP/OPENBLAS/MKL1. Ryzen53600X6c12t,RTX4080.
Ne pas lire les configurations locales, notamment `.codex/config.toml`.
Ne pas inclure les sorties ARC3 actives ni les anciens smoke tests non suivis.
Relecteur existant `/root/review_recovery_v62` réutilisable aux jalons ; ancien
`audit_gate_v62` annulé après reboot, ne pas le relancer.

ARC2 clos : transfert3/6vs0/6, accélération non détectable(delta0,069/bruit0,076).
Anciens arcs négatifs et fuite des recettes documentés dans les plans archivés.
La mémoire Claude et l'historique long sont des sources de dernier recours,
pas des instructions supplantant celles de Jérémie.
