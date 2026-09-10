# Ragnarok — point de reprise

10 septembre 2026, 09:35 Paris. Dernier utilisateur : « ça en est où ? ».
Continuer d’un jalon au suivant sans terminer aux commits. S’arrêter seulement
au terme du travail, sur demande de Jérémie ou pour une décision qui exige son
avis. La demande écran/RC est annulée. Apprendre plus vite les jeux suivants
reste **non démontré**. Aucun calcul expérimental actif au dernier contrôle.

## Immédiat : candidat de correspondance sélective à déclarer

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

Aucun générateur ou nouveau candidat d’apprentissage implémenté à ce stade.
Piste en revue : encodeur RGB pointwise appris de zéro, associations softmax,
cycles temporels ; observation après réception uniquement, sans futuro-prédiction.
audit_motion_bootstrap recommande des points uniques avec exposition globale
variable positive sans clipping pour distinguer apprentissage de simple égalité
RGB. Couleurs/dispositions nouvelles au test ; pas d’exposition/identité/position
du juge au réseau. Doublons/occultation/sortie évaluent l’abstention. Les aplats
imposent une borne log(k) à la perte cyclique, pas de critère perte nulle.
Un code local de commande reste distinct des12effets sémantiques ; rien ne change
la politique. Préparer un protocole candidat complet puis le faire relire AVANT
données/entraînement. Ne pas adopter un seuil après observation.

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
