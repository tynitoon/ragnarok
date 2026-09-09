# Ragnarok — point de reprise

Mis à jour le9septembre2026 vers18:27Paris. Les instructions actuelles de Jérémie
priment sur les pauses historiques. Continuer d'un jalon au suivant ; ne terminer
que sur sa demande, au terme du travail ou pour une décision exigeant son avis.
Objectif : apprendre puis réutiliser pour apprendre plus vite. Une avance initiale
ne prouve pas une accélération. Audit partage d'écran/RC annulé (mauvaise session).

## Lecture courte et preuves

Historique détaillé conservé dans `docs/history/CODEX_HANDOFF_20260909_1811.md`
et celui de16:48. Lire seulement les plans utiles, jamais tous les logs Claude.
ARC2 clos : transfert3/6 contre0/6 ; accélération non détectable(delta0,069/bruit0,076).
ARC4 : aucun apprenant, parser ni gate0 lancé. Vision apprise puis mémoire/transfert,
Atari→NES→16bits→PC. RAM/palettes/référence classique réservées à l'évaluation,
jamais observations ni enseignement du futur agent.

## ARC3 v62 — confirmation active

Commandes revérifiées18:22 : contrôleur20936 `C:/Python314/python.exe -u -m
scripts.recover_confirm_v62`, enfant17340 `scripts.confirm_v61 --lineage 0
--n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume`.
État `v62_confirm_recovery_state.json` running/confirm_s0 ; vérifier commandes
et log réel `craft_v6_out/v61_confirm_s0.log`, puis s1/s2, pas seulement les PID.
Deux mondes entièrement terminés/9 : s0_8300à8326s, s0_8301aprèsG'à18089s.
s0_8302([3,1,0],tiers[2,3,4]) : M4fini20496s, Fafini23090s,
Fbobjectifs3/1finis23849/24613s. Aucun verdict intermédiaire avant toutes les lignées/revue.

Reprise11:37:39Paris après WindowsUpdate06:30/boot06:39. Bras incomplet rejoué
mêmes graines. Interruption2166s facturées ; ancien9366erroné préservé/corrigé.
Budget reprise84742,0902837s ; projection initiale34,3461GPU-h sur42, pas garantie.
Preuves/recovery commitb815be2, `v62_confirm_interruption_20260909/`.
`ARC3_PLAN.md` §12.6 fait autorité, §§13.1–13.5preuves/budget. B6/N9/3lignées/G'2,
store persistant pour tous les bras. Tag gel=a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
Ne modifier aucun des17fichiers scientifiques, des trois lanceurs resume_v62,
confirm_resume_v62/recover_confirm_v62, états ou checkpoints. Aucun second GPUtraining.
Gate/probe PASS revus : M4618/768,R398/768,delta0,286458>3SE0,072446.
Après fin : sources/checkpoints/budget, revue, score gelé, puis figure réelle.
Figure_v61(non gelée) labels/layout corrigés423b770, revueaf94a62 ; rendus
synthétiques seulement, v0/v1échecs conservés, v2PASS, aucun score partiel.
Suivi horaire existant `suivi-ragnarok-arc-3` : ne pas dupliquer.

## ARC4 — instruments et CPU

Les six calibrations brutes et temporelles sont PASS/revues. Sources scellées
immuables. Détails dans `docs/ARC4_<JEU>_CALIBRATION.md` et protocoles associés.
Références CPU précédentes : Pong≈1356, Breakout≈1005, Freeway≈1431,
Seaquest≈1312décisions/s natif. Comparaisons séquentielles sous chargeARC3 ;
aucun débit universel ni gate. Instruments, pas compétences apprises.

Ms17422temporalV2 :33545charges, résultat8e6f930e…/revuee8cfc5c2…,
sources02b5dc7, scellé9ebdfdf, résultate7334f7, relecture/fingerprints c77b860.
Ast17522temporalV1 :34596charges, résultat759d0f42…/revue15b8931c…, a14b07d.
Références brutes fraîches17471Ms/17571Ast :8192frames+5×132=8852charges chacune.
Sources99gelées, préflightefeaa54, captures71ad5e6, scores/revue8a289fe.
Revue commune `v70_visible_reference/result_review_v0.json`, SHAcefb04be…,
99sources/82preuves/64archives, scores exactement reproduits.
Ms rappel97,394%/précision29,463% ; Ast94,090%/92,366%. Juge corrigé source-frame
sans nouvelle calibration brute indépendante : chiffres conditionnels.

FastMs : équivalence40965paires/3613727labels ordonnés/RAMinchangée, committée0cafc2d,
revue `v70_visible_cpu/fast_mspacman_equivalence_review_v0.json` SHA945c47fc…,
102sources/166preuves, commitb9c65a5. Temps décodeur seul135,35→37,08s.
CPU115sources **gelées** àb9c65a5, préflightfinalf1d5486 avant toute géométrie ;
`v70_visible_cpu/preflight_review_v0.json` SHA f5e04e1848d82864c9bc296b86ad97eb2f800954efa0421184f415d2dd8ffad9.
69tests/221preuves ; ne reconstruire ni modifier DLL/build ni115sources.
Géométrie Ms/Ast8192DEV+8192RES chacune PASS hors émulation, commit29cc843.
Wiring18176/18177/18178 : MsPASS3618charges ; AstFAIL265charges au premieract,
Unknown player graphics bank sur décodage anticipéframe0. Échec conservécbc0ef2,
aucune décision rendue ; **aucun retry/substitut ni benchmark Astv0**.
Ms NumPy complet :32768décisions/59,3484s=552,129/s,171364charges, commit7997db0,
résultatSHAe43ecd3a3f055e03bf3e7cfe7d7f0ee857493924f8244f5bc1e197ec606a5542.
Ms **natif complet session33396close**, 580,7928décisions/s,171364charges,
16rapports/digests strictement égauxNumPy ; résultatSHA4851a2cb….
16×2048décisions, graines18000+i, branch128, source/protocoles gelés.
Résultat sous1000 et2000proposés ; aucune nouvelle mesure ni gate. Revue finale à faire.

Ast diagnostic offline a1df7db/0e23739 : cinq RAM0archivées17571déclenchent même garde ;
ce n'est pas la RAM18176nonarchivée, aucune RGB0calibrée. Frame0nonconsommée
aprèsrepeat4 mais nécessaire surrepeat1. Prototype séparé
`deferred_visible_source_v70.py`,8testsPASS, original115inchangé.
Revue préliminaire `v70_asteroids_cpu/deferred_source_review_v0.json` SHA01313bf8…,
118sources/38preuves ; erreurs1frame différées après1observation, ancienéchec0,
ne pas déclarer leurs rapports identiques. Auditofflineconsommation complet :8192frames,
2049groupes(2047×4+2×2), historique/scènes/membres égaux ; session9912close.
Sources/protocole a22ac2b avantaudit, digest ebf68f4886dfd06835cfed54e3f149aefbe86ca4ea23de2f0e1149e299557c34.
Résultats `deferred_source_consumption_v0*`, revuefinale àfaire. Aucune émulation Ast.

## ARC4 — traces et archive, puis apprenant

Trace RGB native **complète PASS** :9cas/26instances/32froids/22604charges
(19158ordinaires+326branches+52×60restores). Sources20gelées87c0507 ;
préflight `v70_rgb_branch_trace/wiring_preflight_v0.json` SHAf0d8a22a…committécbc0ef2
avant18700sixjeux/18711/12/20Pong. Résultats/tentative/console exacte/revue0e23739.
Revue finale `result_review_v0.json` SHA64e82d1b7f6b543ab8db7530771cca11590a0eaab9f03ca5f87dc88401b69902.
SeulSeaquest18700diffnon-nulle ; validité du transport, aucune compétence ME.
Ne modifier aucune des20sources de ce protocole, dont ARC4_BRANCH_TRACE_DRAFT.md.

Nouveau draft **non gelé** `rgb_archive_v70.py` +tests/doc ARC4_RGB_ARCHIVE_DRAFT :
sauvegardes privées20000/2GBpayload, jeu ordinaire continue ; paire sur archive
2T+180(troisrestores), retour au point courant. Pas de retour durableE4 ni RPC.
15testsPASS, revuepréliminaire `v70_rgb_archive/source_review_v0.json` SHAafbe5846…,
11sources, sources7a7fe35/817bb97 ; allocation/calculcompteur protégé avantmutation.
Nouveau transport `rgb_archive_pool_v70.py`,7testsPASS dont2vraisworkerssynthétiques :
plafondtotalpool réparti parworker, RPCfermé, erreurslatchées, cleanup/reportpréservés.
Transport encore àrelire ; aucune modification des11sources déjàrevues.
Aucune nouvelle émulation autorisée par ce prototype.

Suite selon `ARC4_VISION.md` §§2,5,6,11 : parser/tracker/ME/buttons/reader/drives.
Descripteur49floats corrigé(addition), nonimplémenté. `ARC4_ENTRY_BUDGET_DRAFT.md` :
29pairesE1de8frames sur16points COURANTS,3944exp/19720total dans20k, pas16×18.
Traces8framespartageables avecCpréfixe4 ; provenance/points/couverture/unitélag
à figer. MortALE : Pong/Freewaysansvies, régler éligibilité avant gel, pas fauxF1.
Plafond gate0proposé12GPU-h ; aucun lancement avant instruments/revue/gel.
Boxing/Frostbite/Riverraid/Kangaroo **jamais ouverts**. Pas d'apprentissage sur
captures d'instrumentation, aucune RAM/palette/jugement oracle dans le futur agent.

## Environnement et préservation

master, commits ciblés, pas de push ni suppression. Ne pas lire les configurations
locales, notamment `.codex/config.toml`. Exclure sortiesARC3actives/ancienssmokes.
`SETUP.md` : GPU C:/Python314/python.exe Torch2.11.0+cu126 ; CPU
.venv-arc4/Scripts/python.exe Python3.14.3, ALE0.12.1, numpy2.4.4, scipy1.18.1,
pytest9.0.3, pillow12.3.0, sansTorch. Bench OMP/OPENBLAS/MKL=1.
Ryzen53600X6c/12t, RTX408016GB. Pas de réinstallation/reconstruction scellée.
Réutiliser le relecteur aux jalons scientifiques ; audit_gate_v62 ancien annulé.
