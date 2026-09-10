# Ragnarok — point de reprise

10 septembre 2026, 08:19 Paris. Dernier utilisateur : « ça en est où ? ».
Continuer d’un jalon au suivant, sans terminer aux commits. L’apprentissage
plus rapide des jeux suivants n’est PAS établi. Demande écran/RC annulée.
Historique détaillé : docs/history/CODEX_HANDOFF_20260910_0811.md et _0619.md.

## Travail immédiat

ARC3 CLOS9/9, revue conforme3cab686, score INCONCLUSIVE et figure58ea641.
ARC4 fit CNN à voie directe EN COURS, PID12332 depuis08:10:05 Paris.
Aucun nouveau score/calcul ARC3 ; ne pas relancer le fit ARC4.
audit_motion_bootstrap prépare son instrument de revue, attend le résultat
SCELLÉ et le signal root avant décodage. Commits ciblés avec git commit --only.

## ARC3 — clos, INCONCLUSIVE

Neufmondes complets, lignées8300–02/8310–12/8320–22, B6/N9/G2 inchangés.
Contrôleur11880/child21216 absents, arrêt05:50:34,294716Z=07:50Paris.
Résultats6c7f677, revue3cab686 PASS17sources/133preuves/41artefacts sansmodèle.
Sourcefreezea189ea5, autoritéARC3_PLAN§12.6 et§13, aucune relance.

Score gelé exécutéUNEFOIS, figure corrigée423b770/revueaf94a62 exécutéeUNEFOIS,
inspectée bonne ; sorties+doc+reçu58ea641. Δprincipal+.0549321>2se.0527736,
mais lignées0−.0446972/1+.2106509/2−.0011574 :c’est leur condition qui échoue.
Le motif générique conserve une inégalité fausse dans son OR et la phrase
«not distinguishable from zero at this N» ;le compte rendu explique le vrai
motif sansréécrire le code ni le résultat. Tier3+.107648<2se.149103.
Frais–frais t_init−.803866 est calculé surAIRES,pasuniquementb0.

Compte rendu docs/ARC3_V62_CONFIRMATION_RESULTS.md ;sorties
craft_v6_out/v61_confirm_score.json SHA
9210c96d558432510ab8ad8780f98b1a112aae3c4e5a5d2c682dfd5eb5f0b155,
v61_figure.png SHAecc0fa40e287cc0d460f2f417ff53d6f34958a2b1275b5ba1520dd18a5ecf70a,
v62_confirm_publication.json. Review_recovery vérifie maintenant la cohérence
finale du compte rendu JSON-only ;ne pas changer la doc couverte parSHA sans
préserver le premier reçu et faire un addendum si besoin.

Budget38,6756123h jusqu’àclôture pannes comprises ;39,1577527h comptées jusqu’au
reçu de score/figure sur42. ÉchéanceORIGINALE09:10:02Z=11:10Paris nonrepoussée.
Anciennesstates/smokes/notches non suivis àpréserver et ne pas inclure.
Deux incidents et correctionsduvérificateur conservésdanspreuves.

Tier2Δ−.104312,tier3+.107648,tier4+.161460 :le gain des objectifs difficiles
ne remplacepas le verdict global. Δ0tiers3/4 contientpoidsETstoreconservé.
L’aire aprèsb0 peut garder uneavanceinitiale,n’isole pas la vitesse.
Probe618/768vs398/768positif,mais aucuneAUC têteloi des cellules exercées en
confirmation. Symbolique :aucun jeuAtari maîtrisé ni transfertpixelsprouvé.
ScorerSHA75376878619a74655927c652921b668d50eb4daa0bea11fa95dbd4c5fa837238.
FigureSHAcorrigé d8d45dd6931accd26553d3b29030e35cf8fecbdee8edf2ecb916eb74e23e21cd.
L’ancienhand-off avaitunSHAfigureerroné ;gardepréparatoireaarrêtéAVANTPython,
sourcevraimentinchangéconformeàlarevuehistorique. Unseulrenduréel.

## ARC4 — fit direct en cours

Sources75278b2 ; docs/ARC4_STATIONARY_DIRECT_PATH_DRAFT.md et PROTOCOL.json,
ragnarok/learning/direct_response_envelope_v70.py,
scripts/fit_stationary_direct_v70.py. CNN10177+7coefficients RGB1x1 à zéro,
10184paramètres. Même seed764, float32, Adam3e-4/clip1/256visuel+256constant,
12exemples stationnaires anciens, CPU1/1200s dont10s clôture/64MB. Aucun nouveau
tirage, GPU, jeu ou poids appris importé. Critères sur somme seulement :
IoU≥0,25, avantage≥0,10, chacun des2vides FP≤0,001. PartieCNN/direct sauvegardées
pendant3évaluations comptées, diagnostics descriptifs sans forward de plus.
La branche change aussi les gradients et le clip global ; pas de causalité
attribuable au seul accès direct ou aux caractéristiques internes.

Avant Adam, signatures des12tenseurs CNN et sortie initiale/métriques doivent
être identiques au fit négatif historique. Manifeste
ARC4_STATIONARY_DIRECT_INITIAL_IDENTITY.json SHA
aba7be3da5ec1f5416e22bd158d5f5c4f84575542f5d945de0548274c95c1d70.
Le manifeste contient des signatures, pas des valeurs ni poids appris.
Sa création a décodé une fois12tenseurs d’initialisation ancienne,0modèle/forward.
Tests2modèle+5runner PASS6,64s, fixtures seulement ; initialité, parties,
échecs et états préservés. Ne pas les relancer sans modification pertinente.

Préflight948148a PASS119sources/8preuves,15,6685s,0décodage/model/forward.
SHA891f582ead06ef741bfbc72541cefdb0d151e8817055415853ad02cbc2eb4e83.
Dossier craft_v6_out/v70_stationary_direct_fit/. Unique WMI caché horsJob,
launch.py520bd6f, intent/dispatchbe2c380, console confirme detached_from_job.
PID12332, départ06:10:05Z. Reçus/états/logits/probas journaux s’écrivent au fil
 du run. Aucun retry. Attendre état complet et absencePID, sceller fichiers
puis signaler audit_motion_bootstrap pour revue indépendante avant conclusion.

## ARC4 — résultats établis

Premier fit Atari négatif (738,15s CPU1), IoU0,04739, avance0,04249, vides FP
0,5655%/0,1607%, trois clauses échouées. Revu724ab24, figure3d85bc6 et atlas
3fc15b8 vérifiés. docs/ARC4_RESPONSE_ENVELOPE_FIT_RESULTS.md. Validation
conditionnelle19490/19491 INTERDITE par condition échouée. Pas de nouveau fit Atari.

CNN stationnaire même recette négatif744,03s CPU1 : IoU0,00493, avance0,00241,
vides FP0,23214%/0,47024%, 512updates,515forwards. Artefacts19f5ef4,
revue validev1 4c612de SHA c2fe9c270afa098f43c403b3f4818460bd9540993794049cd7df04f1151daa98.
Figure ded76ed vérifiée ; docs/ARC4_STATIONARY_FIT_RESULTS.md.
Validation conditionnelle775 NON EXÉCUTÉE et gardes SHA None restent ainsi.
Son instrumentdef8908 ne doit pas être activé par un autre fit devenu positif.

Affine comparatif complet/revu :7paramètres float64 zéro, Newton amorti7/4
16pas×2bras, Adam256pas×2bras. Résultat3ef0cf9,46,8005sCPU1,544updates.
Newton primairePASS :847/847positifs,IoU/AP1,0FP ; AdamPASS :546/847,IoU0,565774,
AP1,0FP mais cas1pixelentièrement manqué. Même12exemples, aucun nouveau tirage.
Revue0015732,112sources/25preuves/8245événements,0nouveau produit modèle.
Figure fit_v1.png4b79801 inspectée (v0 préservé). Compte rendu
 docs/ARC4_STATIONARY_AFFINE_RESULTS.md mis à jour63ac71b.

Certificat algébrique de famille COMPLET3133ac0,21,9843s, revu18613c4 :
117sources,5coefficients examinés en fractions exactes,0nouvelle image/modèle.
Newtonvisuel seul certifié, minpositif1,3150055/maxnégatif−0,5142525 ; Adamnon.
Domaine déclaré :deuxpassésidentiques,Rbinaire,B=0siR=1,G/Blibressinon.
Tous arrangements dans CE DOMAINE, fonction exacte des coefficients stockés,
pas garantie de tous arrondisBLAS/sigmoid, pas Atari ni transfert.5résultats
publiés sans resélection. Dossier v70_stationary_affine_family/ ; revueSHA
25e591a48725d2b744049ec1c469fae0b9da403f4d48677655bd86cafe836dc1.

## Cap, limites, environnement

ARC4_VISION§2.6 interdit oracle/RAM agent, embeddings bouton/jeu/sprite,
paramètres façonnés à la grille/palette. CoefficientsRGBgénériques autorisés,
mais aucune invariance couleur démontrée. Réservés Boxing/Frostbite/Riverraid/
Kangaroo JAMAIS ouverts. Gate0 non gelé, plafond12GPUh/0B10/STOP inchangé.
Cinq anciens fits mouvement synthétiques1280updates tous négatifs. Le petit
CNN est un diagnostic, pas encore le parser complet P→m→C/ME, l’agent et ses
pulsions, ni le curriculum et transfert N→N+1. Garder ambitions et preuves séparées.

Python C:/Python314/python.exe3.14.3,Torch2.11.0+cu126,np2.4.4,Matplotlib3.10.8.
CPU :CUDA_VISIBLE_DEVICES vide,OMP/OPENBLAS/MKL_NUM_THREADS1.
.venv-arc4/Scripts/python.exe pour ALE, pas de Torch. SETUP.md pour détails.
Ne rien supprimer, pousser ni inclure des configs locales. Ne pas exposer
leurs contenus. Commits--only pour éviter index partagé entre agents.
Automation existante suivi-ragnarok-arc-3 à actualiser après clôture ; ne pas
en créer de doublon. Tâche01a082c5-c85d-7031-95e4-8d432972cb6e.
