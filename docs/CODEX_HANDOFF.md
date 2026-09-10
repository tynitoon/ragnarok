# Ragnarok — point de reprise

10 septembre2026,08:39Paris. Dernier utilisateur : « ça en est où ? ».
Continuer d’un jalon au suivant sans terminer aux commits. Ne s’arrêter que
travail fini, demande de Jérémie ou décision réellement nécessaire. Demande
écran/RC annulée. Apprendre plus vite les jeux suivants reste NON DÉMONTRÉ.
Historique détaillé : docs/history/CODEX_HANDOFF_20260910_0839.md puis _0811.md.

## Immédiat : dernière calibration linéaire locale Atari

Aucun calcul expérimental actif au dernier contrôle. Nouveau candidat
prospectif c3300df docs/ARC4_ATARI_AFFINE_DRAFT.md, instrument79ee2e1 :
scripts/fit_atari_affine_v70.py, tests/test_atari_affine_fit_v70.py,
docs/ARC4_ATARI_AFFINE_PROTOCOL.json.15fixtures PASS5,53s,session46006 FERMÉE.
Aucune archive ni modèle réels ouverts par ces tests. Math affine inchangée.

audit_motion_bootstrap revoit maintenant le code et prépare le préflight
indépendant,96sources/9preuves attendues. Reçu attendu
craft_v6_out/v70_atari_affine_fit/preflight_v0.json,
kind atari_affine_independent_preflight, passed et maps exactes. Il ne lance
pas le fit. Root attend PASS COMMITTÉ puis lance UNIQUE --run,CPU1/CUDAvide.
Budget120s dont10s clôture,64MBsorties ;32updates,67produits/objectifs,
32gradients/Hessiennes/résolutions. Tousseptcoefficients zéro,float64,
Newtonamorti7/4 sans pénalité,16visuel+16constant alternés. AUCUN Adam.
Dataset v70_response_envelopes/targets_v0.npz déjà revu,12ancresAtari/1647pixels
cibles, deux Seaquest vides. Aucun nouveau frame/jeu/RAM/poids appris chargé.

y=Xw :six RGB locaux des deux passés plusbiais,partagés surpixels/jeux.
BCEpondérationancre/classe inchangée ;IoU≥.25/avance≥.1/chaquevideFP≤.001,
seuil.5,toutes12lignes/initial/constant/zéro/trajectoires publiés. Après
clôture : sceller,faire revue indépendante,figure JSON-only et compte rendu.
Sources réellesexécutées :90héritéesducandidatAtari+math/test2+runner/test/doc/proto4.
Ne pas modifier les anciens protocoles ou sources. Tracehashjournal protégé
par try avantpayload ;erreurs/états partiels conservés ;aucun retry.

Review_recovery a examiné la question, avisfavorable aveclimites :PASSprouverait
une suffisance au seul critère de développement ;FAIL16pas ne prouve ni optimum
atteint ni nécessité ducontexte. Modèle/initial/precision/solveurdiffèrentduCNN,
pas une ablation causale. Choix adaptatif aprèsobservation de ce lot déclaré.
Dernier repèrelinéaire de cette série :pas de réglagesenboucle aprèsrésultat.
Aucune validationconditionnelle antérieure réouverte, aucunraccord P→m→C/ME.

## ARC3 clos : INCONCLUSIVE

9mondes B6/G2 achevés07:50Paris ;PIDs11880/21216 absents. Résultats6c7f677,
revue3cab686 conforme17sources/133preuves/41artefacts. Score/figureUNEFOIS58ea641,
PNG inspecté, compte rendu docs/ARC3_V62_CONFIRMATION_RESULTS.md. Dernière
revue de cohérence review_recovery :PASS, aucun changement ni calcul nécessaire.
Δ+.0549321 dépasse2se.0527736 maislignées0−.0446972/1+.2106509/2−.0011574 :
conditiondelignéeéchouée. Le motif générique imprimeaussiune inégalité fausse
et«not distinguishable…»,préservésintacts etexpliquésdansdoc. Aucun nouveau score.
Aireaprèsb0n’isolepasvitesse ;tiers3/4départcontientpoidsETstore.Probe618/768vs398/768
positif,maisaucuneAUC têteloi cellulesexercées enconfirmation. Symbolique,seulement.
Budget38,6756123hjusqu’àclôturepannescomprises,39,1577527hjusqu’aureçuscore/figure,
plafond42h,échéanceORIGINALE09:10:02Z=11:10Parisnonrepoussée. ARC3ESTCLOS.
Sourcesautorité ARC3_PLAN§12.6/13 ;aucune relance. Reçu v62_confirm_publication.json.
ScorerSHA75376878619a74655927c652921b668d50eb4daa0bea11fa95dbd4c5fa837238.
FigureSHA d8d45dd6931accd26553d3b29030e35cf8fecbdee8edf2ecb916eb74e23e21cd ;
ancienhand-offSHAerroné,a causé ungardeAVANTappel,sourceinchangéconfirméaf94a62.

## ARC4 : trois CNN négatifs, affine artificiel positif

AtariCNN :738,15s CPU1,IoU.047393/avance.042491/videsFP.5655%/.1607%,FAIL3clauses.
Revu724ab24,figure3d85bc6/atlas3fc15b8 vérifiés. Rapport
 docs/ARC4_RESPONSE_ENVELOPE_FIT_RESULTS.md. Validation19490/19491nonexécutée.
CNNstationnaire :744,03s,IoU.00493/avance.002409/FP.23214%/.47024%,FAIL3clauses.
Revuevalidev1 4c612de,v0conservé. Figure ded76ed,rapport
 docs/ARC4_STATIONARY_FIT_RESULTS.md. Validation775nonexécutée,gardesSHA None.

CNNdirect10184 :fit2b2de7e complet756,81977sCPU1,512updates515forwards,0erreurs.
InitialCNNstrictementidentiqueetvoie7zérosavantAdam. IoU.00736150/avance.00484067/
vides43et71pixels (.127976%/.211310%) :FAIL3clauses. CNNseulIoU.00524857,
directseulIoU.07214391/AP1mais934et924FPvides. La décomposition ne donne pas la
cause de l’apprentissage,cargradients/clip/biaiscouplés. Aucun nouveau seuil.
RésultatSHAcea34fdcdc5f1de67e6edc595184a6b961770a018700874fc2a3461f5d6b37b5.
Revuef3f98ce,119sources37preuves120lignes9BCE,7,40s,aucun nouveauréseau/forward.
SHA723596fefb4a094c38053d9cd40b019abd57c5f7f69b202b86f51a21e8bef978.
Instrument841e1f7,8fixturesPASS,tousétatsconservés.PID12332absent.
FigureJSON-only sourcepréparée1959ecf puisespacéeavantpremierrendu ;PNGvuBON,
outputs+rapportb0e23cd docs/ARC4_STATIONARY_DIRECT_RESULTS.md. Aucunretry.

Affineartificiel7param :46,8005sCPU1,544updates (Newton16×2etAdam256×2).
NewtonprimairePASS847/847IoU/AP1,0FP ;AdamPASS546/847IoU.565774,AP1,0FP,
mais cas1pixelmanqué. Revu0015732,figurefit_v1.png4b79801BON.
Rapport docs/ARC4_STATIONARY_AFFINE_RESULTS.md. Certificatfamille3133ac0,
revue18613c4 :Newtonvisuelseulcertifiéminpositif1.315/maxnégatif−.514,sur
passésidentiques/Rbinaire/B0siR1/GBlibressinon. Zéroimage/produitmodèle
supplémentaire. Pas degarantieBLAS/sigmoiduniverselle, pasAtari/transfert.

## Contraintes et environnement

La visioncomplète P→m→C/ME,lecteur,pulsions,politiquemémoireetcurriculumresteà
construire. Petitsfits=diagnostics, pas agentqui joue. Cinqanciensfitsmouvement
synthétiques1280updateségalementnégatifs. ARC4_VISION§2.6 :pas oracle/RAMagent,
pas embeddingsjeu/bouton/sprite,niparamètreformegrille/palette. RGBgénérique
encodeapparence,aucuneinvariancecouleurdémontrée. Réservés Boxing/Frostbite/
Riverraid/Kangaroo JAMAIS ouverts. Gate0nongelé,12GPUh/0B10/STOP inchangés.

Python C:/Python314/python.exe3.14.3,Torch2.11.0+cu126,np2.4.4,Matplotlib3.10.8.
CPU CUDA_VISIBLE_DEVICESvide,OMP/OPENBLAS/MKL_NUM_THREADS1. .venv-arc4pourALE,
pasTorch. SETUP.md pourdétails. Pasd’install/push/suppression ;préserverles
configs privés,smokes/notches/anciennesstatesnon suivis. Commits--onlypartagés.

AutomationEXISTANTE suivi-ragnarok-arc-3 adaptéeviaoutil au nom«Suivi Ragnarok
ARC4»,mêmeID/cadencehoraire/tâche01a082c5-c85d-7031-95e4-8d432972cb6e. Prompt
actualisé :ARC3clos,continuerARC4selonhandoff,notifierseulementchangementsutiles.
CompétenceOpenAI Docs lue pourcettegestion,docofficielleconsultee.Aucundoublon.
