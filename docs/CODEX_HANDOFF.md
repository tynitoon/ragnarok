# Ragnarok — point de reprise

Mis à jour le 9 septembre 2026 vers 21:05 Paris. Continuer d'un jalon au suivant,
sans terminer à chaque commit. Arrêt seulement demandé par Jérémie, travail
achevé, ou décision exigeant son avis. L'audit partage d'écran/RC est annulé.
Objectif : apprendre puis réutiliser pour apprendre plus vite. Une avance initiale
ne prouve pas une accélération. ARC 2 : transfert 3/6 contre 0/6 ; accélération
non détectable (delta0,069, bruit0,076). Aucun entraînement Atari ni gate0 ARC4.

Historique : docs/history/CODEX_HANDOFF_20260909_2033.md, puis19:54,19:35,18:11,16:48.
Lire les sections utiles, pas les logs entiers. Calibrations réservées au juge,
jamais à l'agent. Boxing/Frostbite/Riverraid/Kangaroo jamais ouverts.

## ARC 3 v62 — confirmation GPU active

Processus vérifiés20:54 : contrôleur20936 C:/Python314/python.exe -u -m
scripts.recover_confirm_v62 ; enfant22896 scripts.confirm_v61 --lineage1
--n-units3 --b-max6 --num-envs64 --store-persists --g-units2 --resume
(la vraie commande met un espace entre option et valeur).
État running/confirm_s1. s1_8310 terminé à6447s +L ; à21:05, s1_8311 M4 deuxième
objectif terminé7976s. Aucun score/verdict intermédiaire.
Lignée0 complète3mondes/9,25433s : agrégat+14checkpoints validés, committés9deb244.
SHA agrégat3d30c06fe7b6d5be84f042ae26e70c42f0ba70f09610df7eab43cbd7fe66b4ce.
Relire seulement la fin de craft_v6_out/v61_confirm_s1.log pour progrès.

Autorité ARC3_PLAN.md§12.6 ; gel a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
Aucun changement17sources scientifiques/3lanceurs/états/checkpoints. Aucun deuxième
entraînement GPU. Reprise11:37 après WindowsUpdate ;2166s interruption facturée ;
budget de reprise84742,0902837s. Projection34,3461h/42 reste une projection.
Après confirm_complete_review_required : audit9mondes/budget/sources, revue,
score gelé puis vraie figure. Figure corrigée/revue423b770/af94a62 sur synthétique.
Suivi horaire existant suivi-ragnarok-arc-3 : ne pas dupliquer.

## ARC 4 — performance CPU, prochaine exécution préparée

Six instruments bruts/temporels PASS/revus, sources scellées immuables.
Références CPU : Pong1356, Breakout1005, Freeway1431, Seaquest1312 décisions/s.
MsPacman complet : NumPy552,129/s, natif580,793/s,171364charges chacun,
16rapports/digests égaux. Revue40f640d4… àebf4a80 ;115sources/DLL immuables.
Sous1000 et2000 proposés. Mesures sous charge ARC3, pas débit universel.
AstCPUv0 FAILED frame0 Unknown player graphics bank,265charges,0décisions.
Aucun retry/substitut/natifAstv0. Prototype deferred revu offline8192images,
b43a95f5… àebf4a80 ; pas validation d'un rollout natif.

Profilage46226 et audit30037 terminés, sessions fermées. Audit offline complet :
40965paires/3613727labels/24576groupes égaux ; résultat9f137052… àcefdd42.
Image intégrale plus lente31,139→32,634s : garder original. Cachegroupes
47,457→32,243s sur ce composant seulement ; aucun débit total extrapolé.
Revue vector_cached_equivalence_review_v1.json f11f3b69… àdedc57d.
v0 préservée, v1 corrige seulement l'attribution chronologique du document.

Nouvelle chaîne cache SEUL : AUCUN NATIF LANCÉ.
- ragnarok/evaluation/cached_visible_pool_v70.py
- scripts/bench_cached_visible_v70.py
- docs/ARC4_CACHED_CPU_PROTOCOL.json et deux fichiers tests correspondants.
Sources finales1556f12 (après68e2190/f07054d),19testsPASS.
126sources/359preuves ; proofs() readonlyf07054d PASS/runtime exact par reviewer.
Recalcul/revue finale1556f12 EN COURS par review_recovery_v62, avant reçu
craft_v6_out/v70_mspacman_cached_cpu/preflight_review_v0.json.
Préflight committé AVANT wiring ; wiring+revue indépendante committées AVANT
benchmark. Commandes .venv-arc4 -m scripts.bench_cached_visible_v70 wiring/benchmark,
OMP/OPENBLAS/MKL1. Sorties v70_mspacman_cached_cpu/, aucune tentative créée.
Wiring :5instances Ms19176/19177/19178. Benchmark unique natif16×2048,
seeds18000+i/actions/rapports/digests identiques au v0. Pas Ast ni heldout.
Ne pas modifier les126sources pendant exécution/revue. Aucun autre CPU benchmark/
capture/profilage concurrent ; ARC3 reste seul entraînementGPU.

f07054d protège même une allocation échouée de description d'erreur.
Finding1556f12 : wiring perdait une dépense déjà lue si le rapport suivant
échouait ; cache immédiat des rapports/erreurs par nom, témoin2020charges gardées
sans relecture. Reviewer contre-vérifie. Deux consolesv0 déjà hashées par revue
mais ignorées sont désormais committées26be3d5 avec.gitattributes local-text :
bytes inchangés (native0ed08026…,numpy16308a81…).115sourcesoriginelles intactes.
Toutes sessions CPU racine précédentes fermées, y compris proofs58055/81852,
tests57035. Aucun processus CPU natif de collecte/benchmark actuellement lancé.

## ARC 4 — raccord apprenant et limite numérique

Aucune collecte native apprenante. RAM/palettes/références restent hors apprenant.
Modules de base et limites détaillés dans historique20:33 :
parser/P/instances428677paramètres,49floats, revuef6d18381… sourcesd728bae ;
tracker5d25929f… ; borne512pistes/C1d9dc215… ; G/Icb41aa1e… sources37e5d90 ;
PerceptionStream87288ba6… sourcesdd08dbd/doccd0fc78 ; catégorieslocales42c11a5b…
sourcesbf13667. Ne pas modifier ces sources relues ; adaptations séparées.

Chaîne récente :
- OrdinaryRGBTrace :9tests, revue122c7e69… àdedc57d ; départ réel après cold,
 0..4imagesactuelles sans padding. Horloge des images d'action livrées ; charges
 bootstrap/restores exclues. Rupture uniquement si écran brut courant change.
- OrdinaryRGBPool :4tests spawn synthétiques, revueb079d593… à0af4ffc,
 16sourcesdedc57d ; réponses drainées, erreur payée bloque toutpool.
- OwnedRGBCollector :11tests, source0af4ffc, revue6bf43d46…19sources àc3beb18.
 Actions/effets copiés avant paiement, ancrage courant, prise unique, cap64opérations
 sans éviction. Défaut concurrenceb63831d corrigé : verrou public non réentrant.
- OwnedPixelObjectives :12tests, sources379d7bd/doc3954377, revue9e57b87f…
 28sources à68e2190. P surdépart réel/effet enregistré,0..3inconnu/P4 ;
 C4poolé même pour paire8. Inférence rétrospective, jamais rechoisir l'action.
 Erreurs step/branch arrêtent et gardent lot payé ; diagnostic seul n'arrête pas.
- OwnedPerceptionSession :9tests, sourcec3beb18, revuef7d44de2…38sources
 à9e1886d. Iexact8imageslivrées ; T0stable=None ; rupturevisuelle→époque/start0,
 jamais mortALE. Branches horshorloge ordinaire. Batch mémoire atomique par copie
 superficielle sous observe relu atomique ; hypothèse zéro aux observations
 temporelles, P/Ceffetchoisi séparé. Pas validation de lag natif du gate.
- Limite matérielle conservée dans temporal_session_review_v0.json avec recette :
 vrai parser CPU seed764, départ noir, I3,4748144, gradientL2=6,0126643e22 et
 max2,2714049e22 ; carréfloat32 déborde. PASS technique ne valide pas apprentissage.
- GradientGuard :9testsPASS, source9e1886d, PAS encore relu. Norme/carrés/scaling
 float64, gradientsdensefloat32, borneglobale1 proposée ; refusnonfinis, tous
 candidats+reçu alloués avant affectation. Helper sans backward/step/poids/moments.
 Test mécanique du vrai témoin764+unAdam3e-4 SYNTHÉTIQUE : poids/momentsfinis.
 Ses poids de test changent en mémoire ; toujours aucun checkpoint/entraînement
 Atari. Pas optimiseur transactionnel ; clippingàrevoir/figeravantentraînement.

Traces RGB natives anciennes PASS9cas/22604charges,20sources87c0507 et revue
64e82d1b… à0e23739. Une seule divergenceSeaquest ; fidélité, pas contrôleappris.
Archives privées/transport revus synthétiquement afbe5846…/36731edf…, pas nativement,
20000entrées/2Go payload totalpool, pasRSS ; pairearchive2T+180, pasretourdurableE4.

Suite active : préflightcache→wiring→revue→benchmark→revue ; revuegradientguard
ensuite. Raccords ordinairesRGB/modèle validés uniquement en synthétique :
validation native ordinaire à préparer séparément. ME/buttons/reader/drives/S/
policy/ledger/fusion catégories restent absents ; boutons→12effets non résolus.
ARC4_VISION.md§§2,5,6,11 ; budgetentrydraft29pairesE1sur16points,
3944exp/19720total. Gate0≤12GPUh proposé, non gelé. Éligibilité mortPong/Freeway
sansvies à résoudre avantgel. AucunGPUARC4 avant ingénierie/protocole/revue/GPUlibre.

## Environnement et préservation

Master, commits ciblés, pas push ni suppression. Ne pas lire .codex/config.toml.
Exclure .codex/, sortiesARC3actives(v61_confirm.json,s1/s2/états), anciens smokes.
SETUP.md : C:/Python314/python.exe Torch2.11.0+cu126 pour tests Torch SUR CPU,
CUDA_VISIBLE_DEVICES vide,OMP/OPENBLAS/MKL1,torchthreads1. .venv-arc4 :
Python3.14.3/ALE0.12.1/numpy2.4.4/scipy1.18.1/pytest9.0.3/pillow12.3.0,sansTorch.
Ryzen53600X6c/12t,RTX408016Go. Réutiliser review_recovery_v62 aux jalons ;
audit_gate_v62 ancien annulé, ne pas relancer.
