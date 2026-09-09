# Ragnarok — point de reprise

10 septembre 2026, vers 01:05 Paris. Continuer d'un jalon au suivant sans final
aux commits. Arrêt seulement sur demande, travail achevé ou avis indispensable.
Jérémie a passé Astra en Maximum, déjà acquitté ; aucun réglage modifié ici.
L'audit partage d'écran/RC est annulé. Français simple, commentaires utiles réguliers.
Objectif : apprendre puis réutiliser pour apprendre plus vite. ARC2 : 3/6 vs 0/6,
delta 0,069 < bruit 0,076 ; accélération NON démontrée. Aucun entraînement Atari,
checkpoint Atari ou gate 0 ARC4. 1024 mises à jour SYNTHÉTIQUES diagnostiques terminées (768 réseau + 256 champs libres).

Détails antérieurs : docs/history/CODEX_HANDOFF_20260910_0036.md, puis 0014, puis
CODEX_HANDOFF_20260909_2357.md,2338,2314,2232. Lire seulement les sections utiles.
Ne pas charger toute la session Claude, les logs entiers ou les configurations.

## ARC3 v62 — confirmation GPU active, cinq mondes terminés sur neuf

Contrôleur 20936 / enfant 22896 vérifiés à 00:26 : C:/Python314/python.exe -u -m
scripts.recover_confirm_v62 ; enfant -m scripts.confirm_v61 --lineage 1
--n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume.
Lignée 1 démarrée 9 sept 18:41:46 Paris. s0 complet : 3 mondes / 25433 s,
15 JSON committés 9deb244, agrégat SHA
3d30c06fe7b6d5be84f042ae26e70c42f0ba70f09610df7eab43cbd7fe66b4ce.
s1_8310 terminé 6447 s + L ; s1_8311 terminé 13634 s + L ; s1_8312 M4 terminé
16760 s et Fa terminé 20698 s. À 01:03, Fb3 terminé 21590 s / Fb0 terminé
22431 s ; dernier objectif Fb1 puis L restent en cours/à faire.
Aucun score/verdict intermédiaire ; ne pas committer les fichiers actifs.
Autorité ARC3_PLAN.md §12.6, gel a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
17 sources CRITICAL + 3 runners / états / checkpoints immuables (histoire).
Reprise 9 sept 11:37 après WindowsUpdate ; 2166 s interruption facturée,
budget de reprise 84742,0902837 s. Projection 34,3461 h / 42 sans extension.
Après confirm_complete_review_required : revue 9 mondes / budget / sources,
puis score gelé et figure 423b770 / af94a62. Suivi horaire existant
suivi-ragnarok-arc-3 ; pas de duplication, aucun autre entraînement GPU.

## ARC4 — fondations établies, limites conservées

Les six instruments et références sont revus. RAM uniquement dans le juge.
Boxing/Frostbite/Riverraid/Kangaroo jamais ouverts. Débits MsPacman sous les
1000/2000 proposés : original natif 580,7928/s, cache 539,3213/s, gather 523,8992/s.
Aucun gain de chaîne démontré ; anciennes sources, DLL et tentatives immuables.
Diagnostic des archives réelles : 72 pas / 69 P / 1300 ancres I / 6 paires,
0 optimizer / 0 émulation nouvelle / 0 GPU, fini. C : 150 instances propres,
ZERO supervision / perte / gradient sur les six paires. Revue ed4cc04.
Norme max avant clip réellement observée 120,42449 ; l'extrême 1e22 est SYNTHÉTIQUE.

Transport RGB entre Python : 9 cas / 1266 pas / 6 paires / 7454 charges,
images, empreintes complètes et comptes EXACTS de l'ancienne baseline ; aucun
modèle. Résultats ca94418, revue 09cb979 (SHA e93d2932c0ad5973558f51729dc189c99523ec5efbc08fce11fd7a93c06ca287).
Tous workers fermés. Pas de mesure de débit de cette variante ni de test à 16 natifs.
Parser ordonné 4e42567 : 448965 paramètres, paire RGB ordonnée seulement dans d,
anciens f/m/c/s poolés inchangés, effet seulement dans d. Revue 258fae5.
Flux ordonné 258fae5 + propriétaires fd0c704, revue 0b474ab : P4 / I exact8,
mémoire deux écrans propres, contrats avant leurs commandes payantes, coûts et
observations conservés sur échec. Pas d'inférence de cette variante sur Atari.
Anciennes fondations / sources / empreintes détaillées dans histoire 0014/2357.

ME / table de boutons / reader / drives / S / policy / ledger / fusion des
catégories restent absents. UNKNOWN nul ne conditionne pas le parser aux boutons.
P/C/ME ont une dépendance de démarrage encore ouverte. Lag1 n'est pas une preuve
universelle de contrôle direct ; lag2 n'est pas une preuve de contrôle indirect.
Gate 0 NON gelé : entrée / amorçage / coefficients / calendrier / registres et
coûts à résoudre avant GPU ; budgets et critères proposés restent conservés.

## Deux ajustements avec réseau — terminés, NÉGATIFS

Même paire de scènes 16×16, apprentissage et mesure sur ces scènes : pas de
généralisation mesurée. Seuil natif strict : L1 < 0,01171875. Source poolée
commune et futurs opposés distants de 0,046875 ; borne commune 0,0234375.

Original : protocole fb1aea3, préflight 95250f9 AVANT l'essai, 55 sources.
512 updates CPU (256 pooled + 256 ordered), 10 sept 00:02:50→00:03:46,
session 19056 FERMÉE. Résultats baedfb9 ; revue complète 24bc47c :
craft_v6_out/v70_synthetic_motion_fit/fit_review_v0.json
SHA 9d71b8ebe3bcdd6bd35e5cc79b5962ccb9293de252e436f125650f0c8c71ccbf.
Pooled final 0,0252121985 ; ordered final 0,025491314 ; critère NON atteint.
55 sources / 16 preuves vérifiées. Aucune réinférence dans la revue.

Perte à trois échelles (1/3/7) : helper 0611c84, revue 37a4d9e.
Protocole b4e01c2 ; préflight b0f7fea AVANT l'essai, 62 sources / 18 preuves :
026927cbe5eafecc0a9ff501d8a9851c83bbcbd533ce74e5acf7956beb1267fe.
256 updates CPU, mêmes poids vierges / mesures natives initiales exactes,
00:26:37→00:27:24 Paris, session 79982 FERMÉE. Résultats 961db60 :
craft_v6_out/v70_blurred_motion_fit/fit_v0.json
SHA 598226a743bfb404e5068ff74648c9564036b4f5e4be28f8644a2b20beef7bfc ;
attempt 0a5aba4286603bee59a1b10ff67bda7824f600b952879a347a203908dc6d368c ;
console 7ddde7c5055e7682678d2a24ec200d797071cbdfae528a89401465a5524e279d (-text).
Final natif 0,02534569799900055 > seuil ; séparation 0,0005912389024160802.
Exécution complète, résultat scientifique NÉGATIF ; 0 émulation / GPU / checkpoint.
Revue terminée et committée96e8db0 : fit_review_v0.json,
SHA44c79f365535b6ef794b1a383b3efb70ecd51e92ccedd3b4a8f9ae88690e86b6.
62 sources / 22 preuves ; aucun ancien fit réexécuté.

## Champs libres — terminé, négatif et doublement revu

Protocole3fea275 : scripts/check_free_displacement_fit_v70.py et docs correspondants.
Deux champs indépendants nuls 2×2×16×16/1024paramètres ; 256Adamupdates lr0,05,
loss1/3/7, critèreNATIF inchangé <0,01171875. Préflight822af61 AVANTl'essai :
SHA c6fcab02a773f562e5bcebe92e67f51a0a01aea9edbfe5de9631394e119639e3,
67sources/23preuves/49moduleslocaux/4tests. Aucun vrai optimizer dans les tests.
Run10sept00:42:38→00:43:20Paris, travail4,84549s horspreuves ; session65543FERMÉE.
Résultats d6b1d77, craft_v6_out/v70_free_displacement_fit :
fit_v0.json SHA c4e72945902e06430e2cff84bf604d9860f6aaef110bba2a12fd4775f41b9e68 ;
attempt b0eefdc02401d5ec0411d45421938ce6cef34f0926a950ccc82fd03c591605fd ;
console bfc0f0b734f4f02bc95933b6d6a36d923c6502fe50507638867ed26d0759a018 (-text).
Final natif0,013014632277190685 > seuil : FAILglobal. Erreurs parscène [0;0,02602926455438137].
SommesRGB [18;1,9904751777648926]. Champs/coverage/RGBfinaux conservés dansJSON.
Scène0 RGBexact mais déplacementnonuniforme, fondnoirartificiellementdéplacé ;
scène1 contraction des blancs près de(5,7), dilution/collisionsintérieures,
AUCUNEprojectionhorscadre, lumièrezéro sur cible. Paspreuveperception/mouvementvrai.
Revue numérique b6c8868 : fit_review_v0.json,
SHA5def75f0ef5e097e4810435e0b235f272bc966e5f8c1e1119cd09ac813753241,
67sources/27preuves, NumPyreconstruction+coverage BITÀBIT, pertesexactes.
SommeNumPyfloat32vsTorch2ULP différent, images/verdictidentiques, expliquée.
Seconde revue : docs/ARC4_FREE_FIELD_SCIENTIFIC_REVIEW_20260910.md,
SHA965a686a8555bddad2556dd0f09e7f6e548fabeecd26f9001a68714a87636087, bbc8b2d.
Ciblesm des sixfondsdéplacés scène0 seraient≈.9961 : calcul arithmétique,
PASune têtem entraînée. Anciensréseauxnonattribuables fautechampsconservés.
Figure05a0050→bbc8b2d inspectéevisuellement, diagnostic_figure_v0.png :
SHA a5cb583f0fd9c93cfcfc25dbed8f58c29782da61892c40edb0dbea817688bbf2.
Aucun réentraînement ou inférenceréseau dans revue/figure.

## Défaut d'occlusion démontré — anciens scripts inchangés

d4cb325 : tests/test_prediction_occlusion_limits_v70.py (2PASS5,15s),
docs/ARC4_PREDICTION_OCCLUSION_LIMITS.md. Jouetexplicitementopaque3×2blanc,
fondnoirimmobile ; fluxblanc±4/fond0 ⇒ collisionavecnoir, coverage2, gris.5,
L1exact0,01171875. BorneJensen : sumwi≤6, sumwi/(1+wi)≤3, donc mêmeborne
pourfluxblancarbitraire fondimmobile (arithmétiqueréelle ; fond sans texture
nonidentifiabledepuisRGB). Éjecter fondnoirfuturedestinations versbas4 donneRGBexact,
mais ciblem>.99 sur6blancsET6noirs. C'estunedéficienceduréconstructeur surcejouet,
pasfréquenceAtari ni causeuniqueancienséchecs. Aucunoracle envoyéàun modèle.
Revueindépendante bbc8b2d : v70_parser_synthetic/prediction_occlusion_review_v0.json,
SHA6ff374ae3723b8728f77bdfba7401de06aab53f681006b8d41c6cc80d84de8e0,
6sources/2tests/4témoinsNumPyexacts. Sources désormaisrevuesdoncimmuables.

## Prochaine étape — priorités de reconstruction, pas de nouvel entraînement

Opérateurinitial7d14f93 CORRIGÉa78829b AVANTtoutfit :
ragnarok/learning/priority_warp_rgb_v70.py, tests/test_priority_warp_rgb_v70.py,
docs/ARC4_PRIORITY_WARP_DRAFT.md.
-Exp(logpriority), RGB/flow/logpriorityfloat32finis, prio[-20,20]sansclipping ;
 aucune tête/étiquette/futur danscetopérateur. Trousnoirsnonrésolus.
-Findingindépendant : ancienplancherrelatifauMAXprioritéassombrissait RGBblanc
 constant1→.11920929 dèsarrivéeinfime2^-23d'unpixelprio20 auprèsdeprio-20.
 TestaéchouéAVANTcorrection,conservédoc/commithistorique, aucun optimizer.
-Correction : mélangepondéréenFLOAT64, atténuationseulement min(coveragegéométrique/1e-6,1).
 Den=mean_priority*clamp(geometry,1e-6), plusden1pourvides. RGB/geometrysortiefloat32,
 weightedcoveragefloat64, boolfloor. Les8testsincluentcecasetpoidssubnormal1e-40.
-Z0formuleuniforme, pasbitàbitgaranti : tolfixturesRGB2e-7,coveragepondérée1e-6.
 Pasborned'écartuniverselle, aucunrésultatancienréattribué ; débitdoublenonmesuré.
-Sourceprimaire : https://arxiv.org/html/2003.05534v1 (Niklaus/Liu,CVPR2020),Eq13.
 Reprendprincipeopérateurseulement, codePyTorchindépendant,pasCuPy/poids/estimator.
 LeurimportanceEq14/15utilise2imagesconnues+flotpréentraîné ; nonrepris.
 §3.1rapporteéchecZdepuisimageuniquement, doncPASpreuveapprentissagechez nous.
 CVFpdf403 ; arxivhtml/pdf lus. Gardercitationquandonattribuecesrésultats.

Tête6f1d633 : priority_motion_parser_v70.py / testcorrespondant / ARC4_PRIORITY_MOTION_DRAFT.md.
Compositionnn.Module, PASsousclassancienparser ⇒ anciensownersnepeuventl'accepter.
visual=OrderedMotionParser initpremierexact ; outputdataclass(visual,log_priority).
NouvelletêteBlock40→32+Conv1(zéroinit)reçoit6RGBordonnés+embedding32+ownflow2,
borne20tanh(logit/20). Effetuniquementviaownflow, aucunfutur/masque/étiquette.
Zinit0, cinqsortiespoidsvisualexactanciens ; premièrécouchefinalereçoitgradient,
troncprioritéinitialementgrad0àcauseWfinal0 (documenté). 4testsPASS7,97s ;
12testsopérateurcorrigé+têtePASS7,23s. Sessions2800/5369FERMÉES ; zérooptimizer.
review_recovery_v62 ACTIF pourcontre-revuea78829b puis tête6f1d633, nouveauxreçus
attenduspriority_warp_review_v0.json +éventuelpriority_motion_review_v0.json.
Ne pasfairesignerPASSavantfindingrésolu. Committerlesreçusquandvalides.

Aucun nouveaufitdeprioritédéclaré/exécuté. Étapesàpoursuivre : revueopérateur/tête,
puischoisirprotocolesynthétiquePseulavecmêmesbudgets/critèressiutile, et surtout
contratdevisibilité/correspondancesinconnuespourm/C. Ne PASraccorderancienm/C
comme corrigéparleseulopérateur ; laprioritén'estpasprobabilitédevisibilité.
Apprendreprioritésdepuispasséseulementrestehypothèse, toutcommed/C/MEeffets.

## Environnement et coordination

Master, commits ciblés, pas de push / suppression. Exclure .codex/, ARC3 actif
(v61_confirm.json, s1/s2, états, checkpoints), anciens smokes/notch. Ne pas lire
.codex/config.toml. Console ignorée : add -f exact + .gitattributes locale -text.
C:/Python314/python.exe : Python3.14.3, Torch2.11.0+cu126, numpy2.4.4,
scipy1.17.1, pas ale_py. Tests CPU : CUDA_VISIBLE_DEVICES vide, OMP/OPENBLAS/MKL=1,
torch threads1. .venv-arc4/Scripts/python.exe : ALE.12.1, numpy2.4.4,
scipy1.18.1, pas Torch. Pas de réinstallation ni rebuild pendant ARC3.
Ryzen53600X6c/12t, RTX4080 16Go. Toutes sessions racine CPU fermées à01:05 ; ARC3GPUactif.
Revue indépendante explicitement prévue par AGENTS : réutiliser
review_recovery_v62 ; audit_motion_bootstrap terminé, deux revues documentées3fea275/bbc8b2d.
Ancien audit_gate_v62 annulé/pending_init : ne pas réveiller. Ne pas finir sur
un commit ou une attente : poursuivre le travail autorisé.
