# Ragnarok — point de reprise

Mis à jour le 9 septembre 2026 vers 20:33 Paris. Continuer d'un jalon au suivant,
sans terminer à chaque commit. Arrêt uniquement demandé par Jérémie, travail
achevé, ou décision exigeant son avis. L'audit partage d'écran/RC a été annulé.
Objectif : apprendre puis réutiliser pour apprendre plus vite. Une avance initiale
ne prouve pas une accélération. ARC 2 clos : transfert 3/6 contre 0/6 ; accélération
non détectable (delta 0,069, bruit 0,076). ARC 4 : aucun entraînement Atari ni gate 0.

Historique : docs/history/CODEX_HANDOFF_20260909_1954.md, puis 19:35, 18:11 et 16:48.
Lire uniquement les sections utiles. Archives de calibration réservées au juge,
jamais à l'agent. Aucun jeu réservé Boxing/Frostbite/Riverraid/Kangaroo ouvert.

## ARC 3 v62 — confirmation GPU active

Vérifié 20:24 : contrôleur 20936 C:/Python314/python.exe -u -m
scripts.recover_confirm_v62 ; enfant 22896 scripts.confirm_v61 --lineage 1
--n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume.
État running/confirm_s1. À 20:32 : s1_8310 terminé (Fb dernier objectif à6447s,
L=[1,1,1]) ; début s1_8311. Aucun score/verdict intermédiaire.
Lignée0 complète3mondes/9,25433s : agrégat +14checkpoints validés et committés9deb244,
SHA agrégat3d30c06fe7b6d5be84f042ae26e70c42f0ba70f09610df7eab43cbd7fe66b4ce.

Autorité ARC3_PLAN.md§12.6 ; gel a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
Ne modifier aucune des17sources scientifiques ni3lanceurs, états/checkpoints.
Aucun deuxième entraînement GPU. Reprise11:37 après WindowsUpdate, interruption
2166s facturée, budget restant de reprise84742,0902837s. Projection34,3461h/42
reste une projection. Sources/proofs détaillées dans historique19:35.
Après confirm_complete_review_required : audit9mondes, budget, sources, revue
indépendante, score gelé, vraie figure. Figure non scientifique corrigée/revue
423b770/af94a62, uniquement témoins synthétiques. Suivi horaire existant
suivi-ragnarok-arc-3 : ne pas dupliquer.

## ARC 4 — instruments et performance CPU

Six calibrations brutes/temporelles PASS/revues ; sources scellées immuables.
CPU références : Pong1356, Breakout1005, Freeway1431, Seaquest1312 décisions/s.
MsPacman complet : NumPy552,129/s, natif580,793/s ;171364charges chacun,
16rapports et digests égaux. Revue40f640d4… àebf4a80 ;115sources/DLL immuables.
Sous1000 et2000 proposés ; pas de débit universel sous charge ARC3.
AstCPUv0 FAILED frame0 Unknown player graphics bank,265charges,0décisions.
Aucun retry/substitut/natifAstv0. Prototype deferred et équivalence offline8192
images archivées revus b43a95f5… àebf4a80 ; pas validation d'un rollout natif.

Profilage offline terminé (session46226 fermée) ; protocole46f42d1, résultatsb56dfa0.
Audit accélération offline terminé (session30037 fermée), protocole860319d
AVANT exécution, résultatcefdd42 :40965paires/3613727labels/24576groupes égaux.
Image intégrale plus lente au total31,139→32,634s : garder original.
Cache des groupes47,457→32,243s sur ce composant seulement : piste à mesurer
dans une nouvelle chaîne native prospective, aucun débit total extrapolé.
Revue finale vector_cached_equivalence_review_v1.json SHA f11f3b694b2f79b5125d65c5910a9060eac2d82ef30d8aba39ec8e87bc8e84e2,
121sources171preuves ; v0 préservée, v1 corrige seulement attribution chronologique
du document. Reçus committésdedc57d. Pas de nouveau protocole/benchmark natif lancé.
Tous processus CPU de profilage/captures/benchmarks fermés.

Traces RGB natives PASS9cas/22604charges, sources20 gelées87c0507 ; résultats et
revue64e82d1b… à0e23739. Une seule différence non nulle(Seaquest), fidélité seulement.
Archive privée + transport revus en synthétique, pas nativement : afbe5846…,
36731edf… ; sources7a7fe35/817bb97/eaad4c0. Paire archive2T+180, retour au point
courant ; pas retour durable E4.20000entrées/2Go payload totalpool, pas plafondRSS.

## ARC 4 — apprenant, prototypes CPU synthétiques

Aucun poids entraîné, aucune collecte native apprenante, aucune RAM/palette/
référence comme enseignant. Modules détaillés dans historique19:54 et leurs docs.
- Parser/P/instances :428677paramètres,49floats/instance ; revuef6d18381…
  àb9e1a1d, sourcesd728bae. d conditionné12hypothèses d'effet,0=inconnu.
- Tracker : revue5d25929f… ; mémoire4/coast30/rayon8px, affectation atomique.
  Wrapper512pistes et C bilinéaire source : revue1d9dc215… àd6521d9.
- G/I : revuecb41aa1e… à93e23c8, sources37e5d90. I exactlag8, mêmeflux/époque,
  négatifs obligatoires. Collapse G et I stationnaire conservé. Float32/64.
- PerceptionStream : revue87288ba6… àcefdd42, sourcesdd08dbd/doccd0fc78,
 8testsPASS. RGB/trackers sans graphes en mémoire, réinférence poids courants.
 Fenêtre[t−8,t]inclusive9observations ; P tronqué inconnu, pertes finies vérifiées.
- Catégories locales : revue42c11a5b… à1cb9f68, sourcesbf13667,5testsPASS.
 42features/max64, export local exclu héritage, aucune fusion comportementale.

Chaîne RGB ordinaire récemment ajoutée :
- ordinary_rgb_trace_v70.py :9tests/revue122c7e69… àdedc57d. Départ réel après
 possible construction froide,0..4imagesactuelles sans padding. Chronologie
 des images d'action livrées ; charges bootstrap/restore exclues. Rupture mémoire
 uniquement si écran brut courant change au départ ; reset invisible indétectable.
- ordinary_rgb_pool_v70.py :4tests multiprocess synthétiques/revueb079d593…
 committée0af4ffc ;16sourcesdedc57d. Réponses drainées, échec payé bloque toutpool.
- owned_rgb_collector_v70.py :11testsPASS, sources0af4ffc. Actions et hypothèses
 copiées avant paiement, tokens ancrage courant, opération consommable une fois,
 capacity64enattente sans éviction, échecs payés conservés. Revue indépendante
 EN COURS par review_recovery_v62. Finding b63831d : réponses inversées possibles
 avec2threads ; corrigé par verrou non bloquant sur6méthodes publiques, paiement
 et commit compris. Reviewer contre-vérifie le correctif avant reçu.
- owned_pixel_objectives_v70.py :12testsPASS, commit379d7bd, PAS encore relu.
 Propriétaire exclusif du collecteur, verrou pendant collecte ET inférence ;
 P/C calculés sur copies effectives de l'opération consommée, aucun vieux
 PerceptionFrame fourni par appelant. P0..3inconnu, P4direct ; C àframe4poolée
 d'une paire unique au point courant, même si paire8. Inférence rétrospective
 jamais utilisée pour rechoisir action exécutée. Toutes têtes/pertes finies,
 erreurs arrêtent propriétaire et gardent lot payé disponible en copie.
 Ne traite pas encore G/I/tracker ; raccord de chronologie à construire.

Suite active : terminer revue collecteur, revue raccord P/C ; raccorder perception
temporelle à la chronologie livrée sans fausse annotation de mort ni doublonT0.
Préserver toutes sources déjà relues, nouvelles adaptations prospectives.
ME/buttons/reader/drives/S/policy/ledger et fusion catégories encore absents.
Correspondance boutons→effets non résolue ; ne pas injecter boutons bruts auparser.
ARC4_VISION.md§§2,5,6,11. Budget draft :29pairesE1 de8images sur16points,
3944chargesexp/19720totales ; pas16×18. Gate0 proposé≤12GPUh, protocole non gelé,
mort ALE Pong/Freeway sans vies exige éligibilité à résoudre avant gel.
Aucun GPU ARC4 avant ingénierie, protocole, revue et disponibilité GPU ARC3.

## Environnement et préservation

Master, commits ciblés, pas de push ni suppression. Ne pas lire .codex/config.toml.
Exclure .codex/, sortiesARC3actives(v61_confirm.json,s1/s2/états), anciens smokes.
SETUP.md : C:/Python314/python.exe Torch2.11.0+cu126 pour tests Torch SUR CPU,
CUDA_VISIBLE_DEVICES vide, OMP/OPENBLAS/MKL1,torchthreads1. .venv-arc4/Scripts/python.exe :
Python3.14.3/ALE0.12.1/numpy2.4.4/scipy1.18.1/pytest9.0.3/pillow12.3.0,sansTorch.
Ryzen53600X6c/12t,RTX408016Go. Réutiliser review_recovery_v62 aux jalons ;
audit_gate_v62 ancien annulé, ne pas le relancer.
