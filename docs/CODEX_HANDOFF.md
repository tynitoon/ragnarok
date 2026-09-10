# Ragnarok — point de reprise

10 septembre 2026, après l’arrêt technique de 10:26 Paris. Dernier utilisateur : « ça en est où ? ».
Continuer d’un jalon au suivant sans terminer aux commits. S’arrêter seulement
au terme du travail, sur demande de Jérémie ou pour une décision qui exige son
avis. La demande écran/RC est annulée. Apprendre plus vite les jeux suivants
reste **non démontré**. Aucun calcul expérimental actif au dernier contrôle.

## Immédiat : STOP technique, proposition prête à soumettre à Jérémie

La tentative photométrique unique a été exécutée à 10:25:55 Paris et s’est
arrêtée à la garde CPU après la génération. **Zéro initialisation 782, réseau,
forward, gradient, mise à jour ou score de production.** Le masque CUDA vide
laisse le GPU visible sur cette installation. Aucun calcul expérimental actif.

Arrêt et archives conservés en b18719f, fit_v0.json SHA
0dd96f31d2730fdd514d83cb204f6af2d40c79b411ad8a2fc69a3d646729f9f1.
7,4001315 s avant le reçu final ; dossier 3 467 616 octets à la revue.
27 sources et 30 artefacts vérifiés par audit_motion_bootstrap, sans décoder de
tableau : docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_ABORT_REVIEW.md.
4 224 images et 24 vues de contrôle conservées ; aucun marqueur de succès,
checkpoint, journal d’update, réponse ou score du candidat.

Diagnostic CPU séparé, scripts/probe_cpu_cuda_mask_v70.py : deux processus
frais sans tenseur, modèle ou optimiseur ; masque vide → disponibilité CUDA
vraie, masque -1 → fausse, 6,6171842 s. Résultat
craft_v6_out/v70_cpu_cuda_mask_diagnostic/probe_v0.json, SHA
c4a18ac39ec68a098f09cf9eb629e94cb7f0bcc679be11cc0ad707d1d8f56645.
Ne pas refaire ce diagnostic et ne pas confondre visibilité avec calcul GPU.

**Pas de relance autorisée actuellement.** Le contrat gelé dit « Toute erreur,
dépassement ou non-finitude arrête sans recommencer ». La proposition préparée
en b863391 est dans docs/ARC4_PHOTOMETRIC_TECHNICAL_RECOVERY_PROPOSAL.md,
avec scripts/photometric_technical_recovery_v70.py. Elle nécessite l’accord
explicite de Jérémie pour cette exception au STOP, pas une simple automation.
Le wrapper conserve les 27 sources et les données, corrige seulement le masque
à -1 et la destination des reçus, 280 s CPU1, 64 MB sur les trois dossiers.
8+7+280=295 s selon une convention explicite de débit, pas une nouvelle preuve
de clôture de l’arrêt ; la fenêtre continue d’origine est remplacée seulement
si Jérémie l’autorise.

review_recovery_v62 a relu la proposition. Les deux gardes signalées sont
ajoutées : vérification des 27 sources avant le fit et empreinte du manifeste
copié avant/après. Son preflight en lecture a passé, sans tableau décodé ni modèle :
craft_v6_out/v70_photometric_technical_recovery/recovery_preflight_v0.json.
Aucun run/worker/judge de reprise avant accord utilisateur.
La commande run exige --owner-approved-technical-recovery ; ne pas fournir ce
flag sans la réponse réelle de Jérémie. Le dossier n’a que .gitattributes et le
préflight ; aucun octet RGB recopié, aucune intention de run. La réponse finale
doit donner le bilan honnête et demander cette exception technique précise.

## Candidat déclaré et instruments disponibles

La dernière calibration locale Atari est close et négative. Ne pas poursuivre
les réglages de cette série. Interface observation/attribution terminée :
observed_response_targets_v70.py, contrat253a807, correction c4b789f après revue
de l’immutabilité des métadonnées NumPy ; 27 fixtures PASS en3,3302088s,
reçus2b7d08a, rapport22cf0e8 docs/ARC4_OBSERVED_RESPONSE_TARGETS_RESULTS.md.
Stockage bytes/shape/dtype, vue fraîche à chaque accès. RGB par horizon exact,
futurs manquants None, m/C toujours inconnus, zéro poids. Aucun raccord ancien.

Mesure suivante déclarée avant données : docs/ARC4_SELECTIVE_CORRESPONDENCE_DRAFT.md.
Scorer selective_correspondence_score_v70.py relu, 17 fixtures arithmétiques
PASS en5,2088419s, aucune image ni modèle. Rapport et reçus dans
docs/ARC4_SELECTIVE_CORRESPONDENCE_SCORE_RESULTS.md et
craft_v6_out/v70_selective_correspondence_score/. Les deux budgets de fixtures
sont distincts et clos, pas des résultats scientifiques de candidat.

Contrat :5strates256scènes IID chacune,1requête juge invisible au modèle par scène.
9exigences à alpha=.05/9 :pour2visibles,CPsup erreur|admis≤.05,CPinf couverture
correcte≥.5,gain observé vsinitial≥.1 ET binomial apparié p≤alpha ;pour3nonattribuables,
CPsup admissionindue≤.05.4témoinsoccultationdéterministesenplus. Archiver toutes
les admissions AVANT juge et les1280verdicts. Cela ne certifie ni validitédense,
ni variabilitéentreentraînements,nirisquesaprèsintersection des admissions m/C.

Candidat déclaré avant toute donnée dans
docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_DRAFT.md (f220669), puis amendement
ARC4_PHOTOMETRIC_CORRESPONDENCE_AMENDMENT_1.md (c3eece7). Les deux agents ont
relu la méthode : encodeur partagé 1×1, 3→32→32→16, SiLU et L2, 1 712 paramètres,
cycles de marches softmax à température 0,07. Admission aller-retour ≥0,90 ET
similarité aux extrémités ≥0,95. Raw seulement ; pooled inconnu sans réseau,
garde structurelle qui ne compte pas comme abstention apprise. Les aplats ont
une perte irréductible, et un faux cycle peut être cohérent.

Implémentation et vérifications prospectives terminées, tentative arrêtée :
ragnarok/learning/photometric_correspondence_v70.py (socle e062d6e),
ragnarok/evaluation/photometric_scenes_v70.py,
scripts/photometric_common_v70.py, photometric_worker_v70.py,
photometric_correspondence_v70.py et judge_photometric_v70.py.
Tests tests/test_photometric_correspondence_v70.py ; protocole JSON dans docs/.
Générateur relu conforme : 8×8, quatre points, couleurs nouvelles et exposition
globale float64 puis rint ; 128 clips visibles pour apprendre, 1 280 clips juge.
Graines 19492, SeedSequence19493, initial782 ; fixtures55103..55105 distinctes.
Pas de position, strate, identité ou exposition du juge au réseau.

Deux processus distincts : génération et archives RGB/juge, puis fit qui ne
décode que les quatre NPZ RGB à membre unique. 512 Adam, mini-lots4 fixes,
initial exact conservé puis évalué après entraînement ; témoin RGB normalisé
descriptif. Attendus si complet : 674 appels appris, 13 848 images encodées, 81 associations témoin,
4 556 bmm explicites de forward. Les opérations internes de backward/convolution
ne sont pas comptées comme ces bmm. 4 224 nouvelles images, 24 vues de contrôle.

Ancien lanceur : scripts.photometric_correspondence_v70, maintenant arrêté.
Ne pas refaire ses phases. Un éventuel juge de reprise exige d’abord une revue
technique indépendante fit_review_v0.json liée au SHA d’un fit complet.
Dossier craft_v6_out/v70_photometric_correspondence/ : 21 fixtures PASS en
4,5945252 s murales (pytest 3,10 s), un backward, zéro mise à jour ; reçus et
clôture positive 742a9cb. Préflight conforme de 27 sources et runtime, 011d232.
Implémentation 1df8e81, frontière de clôture cb7f55e ; génération terminée,
initialisation782 jamais effectuée. Ne pas refaire les fixtures.
Budgets : fixtures 60 s cumulées, un backward maximum, zéro update ; production
300 s murales incluant imports/génération/fit/inférences, réserve15 s, CPU1,
64 MB ; un juge séparé60 s, un seul score, aucun nouveau réseau ou RGB.

Revues du lanceur : corrections prospectives pour sauvegarde par lot, événements
fsync avant/après les updates, bornes du travail interrompu, préflight lié à ses
octets, admission des écritures sous plafond, clôture indépendante des erreurs.
audit_motion_bootstrap a validé le dernier delta avant lancement. Juge relu :
contrôles exacts avec deux branches chacun, empreintes et runtime recontrôlés.
Un score utilisable exige reçu complet ET marqueur positif de clôture lié à son
SHA (un reçu complete seul ne suffit pas), sans marqueur d’échec. L’amendement2
précise que seuls le dernier lien atomique et la console sont hors du temps
scientifique ; payloads, fsync, relectures et SHA sont dedans. Aucun calcul après
publication. L’arrêt actuel ne possède aucune clôture positive de production.

Après propositions scellées et revue : conserver les 1 280 verdicts avant score,
les réponses aux huit clips de contrôle, les rangs initial/final avant abstention
et les coïncidences quantifiées. Garder tous les résultats. Aucun raccord m/C,
ME ou changement des douze effets sémantiques autorisé automatiquement.

Source primaire consultée : https://arxiv.org/html/2006.14613v2 (§2,annexesA/C),
https://ajabri.github.io/videowalk/ et READMEgithub ; aucun poids/dataset téléchargé.
Il s’agira d’une adaptation, pas d’une reproduction des résultats du papier.
Sources locales utiles : ARC4_VISION §2.1/2.2/2.6/6, VISIBILITY_UNKNOWN_DRAFT,
BOOTSTRAP_LIMITS, LOCAL_COMMAND_BOOTSTRAP_DRAFT. Les deux agents existants
audit_motion_bootstrap et review_recovery_v62 sont disponibles pour revues ciblées.

## ARC 3 clos : INCONCLUSIVE

Neuf mondes B6/G2 achevés à 07:50 Paris ; anciens PIDs absents. Résultats
6c7f677, revue 3cab686 conforme, score et figure calculés une fois en 58ea641.
Compte rendu : docs/ARC3_V62_CONFIRMATION_RESULTS.md, vérifié indépendamment.
Δ moyen +0,0549321, 2 se = 0,0527736 ; deux lignées sont non positives
(−0,0446972 ; +0,2106509 ; −0,0011574). Ce dernier critère impose le verdict.
Le motif générique du score contient une inégalité fausse : conservé intact et
expliqué dans le rapport. Ne pas recalculer ou réécrire ce score.

La sonde de connaissances est positive (618/768 contre 398/768), mais ne prouve
pas une accélération. L’aire après b0 peut conserver une avance initiale.
Budget comptabilisé : 38,6756 h à la fin des expériences, 39,1578 h au reçu de
publication, sous 42 h. Source d’autorité : ARC3_PLAN §12.6/13. Aucune relance.

## ARC 4 : bilans clos et conservés

| Diagnostic | Résultat établi | Rapport |
|---|---|---|
| CNN sur douze ancres Atari | IoU 0,04739, trois critères échoués | ARC4_RESPONSE_ENVELOPE_FIT_RESULTS.md |
| CNN sur contrôle artificiel stationnaire | IoU 0,00493, trois critères échoués | ARC4_STATIONARY_FIT_RESULTS.md |
| CNN avec voie RGB directe, même contrôle | IoU 0,00736, trois critères échoués | ARC4_STATIONARY_DIRECT_RESULTS.md |
| Affine sur contrôle artificiel | Newton : IoU 1, zéro faux positif ; certificat limité à la famille déclarée | ARC4_STATIONARY_AFFINE_RESULTS.md |
| Affine sur douze ancres Atari | IoU 0, aucun pixel cible détecté au seuil 0,5 | ARC4_ATARI_AFFINE_RESULTS.md |

Tous ces rapports sont dans docs/. Figures inspectées et revues indépendantes
conformes. Les petits fits sont des diagnostics, pas un agent qui joue.

Dernier fit Atari : 4,6822162 s CPU, 32 mises à jour, 67 produits/objectifs ;
résultats b08e0d2, revue b64bae7, rapport/figure 875c443. Préflight v0 arrêté
avant décodage sur ancien schéma de résultat, conservé ; correction 511ce8d,
préflight v1 deecfd3 conforme. Aucun réessai de fit. Résultat :
craft_v6_out/v70_atari_affine_fit/fit_v0.json, SHA
b168939f7fb3c4fcc81684785de64b49d79a24a07f4500b56e76d6437ac1f121.
Revue SHA b6c2a748f962e42a77717b846f4d23b92f9973d43599bb5a2f1b6a0a7e8bc732.

Les anciennes validations conditionnelles 775 et 19490/19491 restent non
exécutées. Aucun seuil ou nombre d’étapes ajusté après observation. Le contrôle
artificiel positif ne valide ni Atari, ni correspondance, ni m/C, ni ME.

## Contraintes et environnement

La chaîne P → m → C/ME, lecteur, pulsions, politique, mémoire et curriculum
reste à construire et valider. Cinq anciens fits de mouvement synthétique sont
aussi négatifs. Pas de RAM ou d’extracteur oracle dans l’agent, pas de poids
préentraînés téléchargés, ni d’embeddings par jeu/bouton/sprite.
Boxing, Frostbite, Riverraid et Kangaroo sont réservés et jamais ouverts.
Gate 0 non gelé ; 12 GPU-h, zéro 0-B et règles STOP inchangés.

Python principal : C:/Python314/python.exe 3.14.3, NumPy 2.4.4,
Torch 2.11.0+cu126. CPU : CUDA_VISIBLE_DEVICES vide et OMP/OPENBLAS/MKL à 1.
.venv-arc4 pour ALE, sans Torch ; SETUP.md donne les environnements.
Pas d’installation, suppression ou push. Préserver les fichiers locaux privés
et les sorties non suivies des anciens travaux. Commits ciblés avec --only.

Automation existante suivi-ragnarok-arc-3 : nom « Suivi Ragnarok ARC 4 »,
cadence horaire, prompt actualisé à cette suite et notifications seulement
aux changements utiles. Aucun doublon. Historique détaillé dans
docs/history/CODEX_HANDOFF_20260910_0839.md puis _0811.md ; les rapports actuels
priment sur ces instantanés historiques. Ne pas charger tous les anciens logs.
