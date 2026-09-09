# Ragnarok — point de reprise

10 septembre 2026, vers00:14 Paris. Continuer d'un jalon au suivant sans final
aux commits. Arrêt seulement sur demande, travail achevé ou avis indispensable.
Jérémie a passé Astra en Maximum, déjà acquitté ; aucun réglage modifié ici.
L'audit partage d'écran/RC est annulé. Français simple, commentaires utiles réguliers.
Objectif : apprendre puis réutiliser pour apprendre plus vite. ARC2 :3/6 vs0/6,
delta0,069<bruit0,076 ; accélération NON démontrée. Aucun entraînement Atari,
checkpoint Atari ou gate0 ARC4.512updates SYNTHÉTIQUES diagnostiques ont été faits.

Historique détaillé : docs/history/CODEX_HANDOFF_20260909_2357.md puis2338,2314,
2232,2200,2132,2105,2033. Lire seulement les sections utiles. Ne pas charger la
session Claude complète, les logs entiers ou les configurations locales.

## ARC3 v62 — confirmation GPU active, cinq mondes terminés sur neuf

Contrôleur20936/enfant22896 vérifiés23:32 : C:/Python314/python.exe -u -m
scripts.recover_confirm_v62 ; enfant -m scripts.confirm_v61 --lineage 1
--n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume.
s1 démarré9sept18:41:46Paris. s0complet3mondes/25433s,15JSONcommittés9deb244,
SHAagrégat3d30c06fe7b6d5be84f042ae26e70c42f0ba70f09610df7eab43cbd7fe66b4ce.
s1_8310terminé6447s+L ;s1_8311terminé13634s+L ;s1_8312M4terminé16760s,
Fa3à17843s/Fa0à19600s à00:13. Lire tailcraft_v6_out/v61_confirm_s1.log.
Aucun score/verdict intermédiaire ;s1/Fa/Fb/L encoreen cours. Ne pascommitterlive.
AutoritéARC3_PLAN.md§12.6,gel a189ea5bb0f9c8b7e9016a50aefc1cbca603426f ;17sources
CRITICAL+3runners/états/checkpoints immuables. Reprise9sept11:37aprèsWindowsUpdate,
2166s interruptionfacturée, budgetreprise84742,0902837s ; projection34,3461h/42
n'étendpasdeadline. Aprèsconfirm_complete_review_required :revue9mondes/budget/
sources puis scoregelé/figure423b770/af94a62. Suivi horaireexistantsuivi-ragnarok-arc-3,
pasduplicate. Aucun autreentraînementGPU. Runners/figures/preuves danshistoire.

## ARC4 — références, débits et premiers diagnostics conservés

Six instruments bruts/temporels revus, sources scellées. Jeux réservés Boxing,
Frostbite,Riverraid,Kangaroo jamais ouverts. RAM uniquement dans le juge externe.
Débitsréférence :Pong1356/Breakout1005/Freeway1431/Seaquest1312décisions/s ;
MsPacman originalnat580,7928/s,16rapportségaux/171364charges,revue40f640d4…ebf4a80.
AstCPUv0FAILEDframe0/265charges, aucunretry/natifsousmêmenom. Astdeferredarchivesrevues.
Cachegroupes539,3213/s,revue158d1dee…fb078ef ;gatherdécodeur523,8992/s,
revue232335f5…0afd78d ;aucungainchaînedémontré, sous1000/2000proposés. Sources,
DLL,benchmarks figés. Gainscomposantsoffline ne sontpasdesgainschaîne. Histoire2357.

BaselineRGBordinaire877be12/a307984/e5b969c :9cas/18instances/14908charges,
9NPZ/330arrays/72positions+6branches. FULL ordered_trace_sha256 parcas.
Wiring5db50e4ee27fedf0ec77a62fab9c13efd309749f7a127e47d73ec4f56ae886af,
revue8b4105ce479e7d990293a89ce8ea3789506600729f3e1f721dbf59653f04a668.
Originalparser/P/instances/tracker/C/G/I/catégories/collector/objectives/session/
gradientguard etentry29paires tousrevus ; sourcespreuves détailléeshistoire2314/2232.
Entry29paires/16points20k/.2→19720charges dont3944exp/58restores, pascontrôleurexécutant.
ME/tableboutons/reader/drives/S/policy/ledger/fusioncatégories restentabsents.

DiagnosticCPUréellesarchives ca0ede1/ed4cc04 :72pas/69P/1300Iancres,6paires,
0optimizer/0nouvelleémulation/GPUfalse,poidsinchangés. C :150instancespropres,
ZERO supervision/perte/gradient aux6paires. Paspreuvevision/Cutile/apprentissage.
Résultat610ef8a238449939290efa6ad3cbdb2bfe886cbfc5efddf032113540fd85971d,
revue574e1881b67bb1e5e07c8f05c9433533eb09abdbd30507a37465cb5082eac9aa.
Lecturearithmétique00:04 :maxnormavantclip120,42449 (Seaquestindex1),pasréinférence.
Ne pasconfondreavecextrêmeSYNTHÉTIQUEI6,0127e22,histoire. Normeguardnecréeaucuneinformation.

## Transport explicite entre Python — natif terminé et revu

Codec/serveur/client fadaacc/16dc3ab/6e0e4a7,revue source97a3fc7d…4e42567,
21sources/3preuves/27tests. JSONtypé+RGBbytes,sanspickle ; commandescourtesfermées,
1processdirect/worker,3threadsbornés,HELLOexactavantINIT,deadvielowerbound+stopglobal.
Findinginitial601chargesécraséesparrapportNone corrigé6e0e4a7 ;aucunretrypayant.
Pasgarantieenfantaprèsmortbrutaleparent ;futurefinallycleanupnécessaire.

Protocole98d6919 :scripts/check_isolated_rgb_native_v70.py et
ARC4_ISOLATED_RGB_NATIVE_PROTOCOL.json,8tests,46sources/26preuves. Préflight
f4362b1306a1aee45181e0d6f0f37b371842bf9a63f6e4032aa45e77641a482c,
committé8f2eade AVANTrun23:43:51→23:44:51Paris9sept. Résultatsca94418 :9cas,
1266pas/6paires/7454charges/9constructs, images/digests/métadonnées/comptes EXACTS
baseline. Même19200/19211/12/13déclarés, pasnouvellepopulation ;0model/Torchimport/GPU.
Dossiercraft_v6_out/v70_isolated_rgb_native :
wiring_v0SHA81e366e2ec82dc9f3b888429a8c7c878216d067cfad3cec7b46e4b7de75951cb ;
attempt90897ffdef434eacdc12d95e8c073a16b3554a790677e894abc4e99dff60c3d4 ;
consoleb4d9fb52fa1c7ba628953f5fd91d2d2b7e9d9049c3e281629ec5500f8ba6babd (-text).
Revue09cb979 :wiring_review_v0.json,
SHAe93d2932c0ad5973558f51729dc189c99523ec5efbc08fce11fd7a93c06ca287 ;46sources/30preuves,
7454=5058ordinaires+1580froids+96exp+720restores ;330anciensarraysvérifiés.
Session86381FERMÉE,0serveur/runner. Pasclaimnatif16workers/T0/débit/vision.

## Modèle ordonné et propriétaires — sources revues, pas Atari

Contre-exempleMAX :sources4→5/5→4identiques,cibles4framesopposées,distanceL1.046875,
borneprédictioncommune.0234375. OrderedMotionParser4e42567 garde f/m/c/s/pyramid
poolés, ajoutebrancheapprise6→32RGBordonnés, dconcat32pool+32motion+32effet.
448965params,effetuniquementd,Pgradientbranche ;ancienparser428677inchangé.
Revue138a3049877c38636a584cfa0de065c1c28d461aefcc5fa5b203e87a8b4b692f,
v70_parser_synthetic/ordered_motion_review_v0.json,258fae5,26sources/3preuves/8tests.

Fluxordered_perception_stream_v70.py258fae5 +ownersowned_ordered_perception_v70.pyfd0c704 :
contratexactparser/float32finis/deviceunique/HW>=8avantleurscommandes. P4/Iexact8,
histoire2RGBuint8sansgraphes+trackingcopy,max9temps ;Irecalculeparsercompletavecpoidscourants.
TemporalUNKNOWN ;P/Chypothèseenregistrée. Méthodespayantes/lock/stop/rétention
héritéesanciensownersinchangées ;Cbranchn'âgépaslive. _pendingpeutgardergraphes
suréchecmaislotborné/ownerstoppé. Callerfermecollectorsiéchecconstructeur.
Revue0b474ab :ordered_temporal_review_v0.json,
SHAbed82176921166623ce4f289b1b9c595121e91dd9af7f63fd71727bfd07da270,
51sources/12preuves/13tests+témoinsmultiworker1208/2records,T1/3/4,T0x2coût864.
HéritageCounterfactualParsern'impliquePAScompatibilitéoldsignature :nepasinsérer
candidatdansanciensownersquipeuventl'accepterpuiséchoueraprèspaiement.

## Fit synthétique P original — NÉGATIF, revue résultat en cours

Protocole86eeb9f CORRIGÉfb1aea3 (scripts/__init__.pymanquantfermeture,avantfit).
scripts/check_synthetic_motion_fit_v70.py / ARC4_SYNTHETIC_MOTION_FIT_PROTOCOL.json.
55sources/12preuves/4tests, préflight75a501616c42935eb81cdc4a19990305abc4ab130e6470532d2a037254637253
committé95250f9 AVANTrun10sept00:02:50→00:03:46Paris. Session19056FERMÉE.
Résultatsbaedfb9,craft_v6_out/v70_synthetic_motion_fit :
fit_v0SHA9d049a632816855b5dd2497681c9ff900bdbe04d7d8efbb68bb0cbb56514b82f ;
attempt106b1cde214def9d70e7d2713e9c218ad1849e82c204875b0d090d13fa739797 ;
consolecc794bfb70970c96ac2e89670bb5105e457875c3e0a13ae9cfdcc48dabef875c (-text).
Deuxscènesseulement, samefit/mesure,pasgénéralisation. pooledpuisordered,seed764,
256updateschacun,batch2,Adam3e-4/maxnorm1/PimageL1seule.512updatesCPUfinis,
aucunAtari/GPU/checkpointretenu. Pooledfinal.0252121985>=borne.0234375 ;ordered
.025491314>seuil.01171875 :critèreNONatteint,exécutionnumériquecomplète.
InitialorderedSHA dc033e685da63a838b5e09dc61e80f9a64c445fac9033700cccee224ed90905a,
finalc72e2df6901ff17da4976f49db7ec57e677fefc4ee245c4e1f8df6e55669285f.
Le marquageoptimizer_step_in_progressestenmémoire,persistéfinallysurexception ;
arrêtdurstartedincertain,noretry. Tous55sources désormaisimmuables.
Review_recovery_v62 ACTIF sur résultat ; reçufutur fit_review_v0.json,
aucunrejeu/optimizer autoriséparrevue. COMMITTERreçuavantpréflightcandidate.

## Prochaine expérience préparée : perte blur, pas encore de mise à jour

LimiteP f47f214 :uniformdx±.25/±.5 carrés4/5→8/9,L1.046875/gradientdx0 ;dx4L1=0 ;
jetersourcehorsécranL1.0234375. GradientDENSEpeutêtrenonnul(.0302734témoin),pasclaim
Atarifréquence/apprentissageimpossible. ARC4_BOOTSTRAP_LIMITS gardeaussiboucleeffets/C/ME.

Candidateblur5047c5f CORRIGÉ0611c84,préparéeAVANTrésultatfitoriginal :
blurred_prediction_loss_v70.py,8tests,meanRGBL1largeurs1/3/7,stride1/padzero,
filtreDANSLOSSseulement,pasfrontend/mask/flowlabel ;future.requires_gradREFUSÉ,
calculfloat32,mêmeinputfloat64. OwnRGBfutureprovenanceresteappelant.
Revue37a4d9e :blurred_prediction_review_v0.json,
SHA8556e81cc7121bf5f8e8246f496e0517bd5325b982f865e9642fdf851f23bad1 ;9sources/8tests,
gradienttranslationorienté±.00396825confirmédifférencefinie ;pasoptimizer/Atari.
RevueMEtempsmêmecommit :agency_timing_review_v0.json,
SHA2056a3f6b8c3d29fecc00757229244418f37166cc88e6cca96ca7941b0bf9c9a ;11sources/2tests.
SimulationcausaleAcommandépuisBcausédansmêmeframe→lag1aux2 ;stickyretardeAàlag2.
PasfréquenceAtari/verdictheuristiqueMEcomplet. SourcescandidatsREVUESdoncgelées.

b4e01c2 :scripts/check_blurred_motion_fit_v70.py,
tests/test_blurred_motion_fit_protocol_v70.py5PASS17,91s,sessions89797/93254FERMÉES,
docs/ARC4_BLURRED_MOTION_FIT_{DRAFT.md,PROTOCOL.json}. Aucunoptimizercandidat.
-orderedseul256updates,mêmespoidsvierges+mesuresinitialesstrictes,scènes/seed/Adam/
 norm quebaseline ;seuleloss1/3/7change. PASrejeudesanciensbras.
-CritèreTOUJOURSNATIVEimageL1<.01171875 ;métriquesfiltréesrapportéesséparément.
 Premieréchectestcroiséfloatglobalvsparscène3,73e-9corrigétol1e-8, pascritèrechangé.
-CPU300scoopératif,0native/GPU/realarchive, marqueurstepincertain,noretry.
-proofreçoitbase.proofs()+revuesfitoriginaletblur ; proofs()pasencoreexécuté car
 revuerésultatoriginalenattente. Sourceclosureàauditer, nepasdevinerlenombre.
-Préflightindépendantfuturcraft_v6_out/v70_blurred_motion_fit/preflight_v0.json,
 kindblurred_motion_fit_independent_preflight/exactprooffields. COMMITTERAVANTrun.
 Reviewerconnaîtprochaineétape ; finaliserreçuoldfitpuisrevueceprotocole.
-Ensuiterununeseulefois,préserverconsole/attempt/result,revuenumériquepuisjalonsuivant.

## Environnement et règles

Master/commitsciblés,paspush/suppression. Ne paslire .codex/config.toml. Exclure
.codex/,ARC3live(v61_confirm.json,s1/s2/états/checkpoints),ancienssmokes/notchv62.
C:/Python314/python.exe :Torch2.11.0+cu126/scipy1.17.1,numpy2.4.4,pasale_py.
TestsCPU :CUDA_VISIBLE_DEVICESvide,OMP/OPENBLAS/MKL1,torchthreads1 ;pasGPUARC4.
.venv-arc4/Scripts/python.exe :Python3.14.3/ALE.12.1/numpy2.4.4/scipy1.18.1,
pytest9.0.3/pillow12.3.0,sansTorch. Aucunereinstallation/rebuildenvironnement.
Ryzen53600X6c/12t,RTX408016Go. Réutiliserreview_recovery_v62 pourrevuesmajoritaires
explicitementprévuesAGENTS ;audit_gate_v62ancienannulépending_init,nepasréveiller.
TOUTESsessionsracineCPUfermées à00:14 ;ARC3GPUtoujoursactif. Ne pasfinirdansattente,
poursuivredèsreçupreflight. Anciennespreuves/commandesdanshistoire2357/2338/2314.
