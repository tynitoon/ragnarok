# Ragnarok — point de reprise

10 septembre 2026, 05:07 Paris. Continuer entre jalons sans terminer aux commits.
Dernier utilisateur : « ça en est où ? ». L’accélération de l’apprentissage
sur les jeux suivants n’est pas établie. Écran/RC annulé : mauvaise session.

## ARC 3 : reprise active

Contrôleur 11880, enfant 21216, démarrés le 10 septembre vers 03:02 Paris.
État actuel : craft_v6_out/v62_confirm_second_recovery_state.json.
Au contrôle de 05:05 : processus présents ; M4 de s2_8321 terminé, Fa en cours.
Sept mondes sur neuf sont complets depuis 04:43 ; s2_8321 est en cours. Le fichier
v61_confirm_s2.json contient le septième ; l’agrégat reste à 6 jusqu’à la fin
de la lignée. M4 du monde s2_8320 avait été conservé.
Aucun score partiel. Les anciens états « running » sont historiques.
Vérifier les processus réels et un tail borné du journal v61_confirm_s2.log.
Ne pas afficher le gros état JSON entier : sélectionner status/stage/PID.

Échéance ORIGINALE : 10 septembre 09:10:02Z = 11:10 Paris. B6/N9/G2,
64 environnements et budgets inchangés ; pannes et investigations consomment
la même fenêtre. Réserve de clôture de 30 s ; surveillance UTC + monotone.
Ne pas utiliser l’ancien elapsed_seconds pour réattribuer du temps.

Sources de reprise f36971b puis 7acff3e ; revue 3d5ace9, 34 tests PASS.
Ne modifier aucun des quatre fichiers recover_confirm_after_interrupt_v62,
launch_detached_confirm_v62 et leurs deux tests pendant cette exécution.
Revue : craft_v6_out/v62_confirm_second_recovery_review.json (9bc32b78…).
Lancement WMI caché hors Job Object, reçu durable avant autorisation GPU.
Originaux des reçus stables dans craft_v6_out ; copies exactes committées
avec l’incident dans v62_confirm_interruption_20260910/ (ca3b24e).
État et journaux actifs restent hors commit. Aucun autre calcul GPU.

Incident du 10 septembre à 02:26 Paris : anciens PID 20936/28140 disparus,
démarrage Windows inchangé, cause inconnue. Traces préservées f36971b.
Voir docs/ARC3_SECOND_INTERRUPTION_RECOVERY.md et ARC3_PLAN.md §12.6.
Après neuf mondes complets : revue indépendante coûts/sources/résultats,
puis score gelé et figure 423b770 (revue af94a62). Pas de relance aveugle.

## ARC 4 : survey terminé, revu et figuré

216/216 comparaisons, six jeux, deux ancres et dix-huit commandes chacune.
Collecte close le 10 septembre à 03:34:52 Paris, ancien PID 20816 absent.
157 530 images facturées, comptes complets, 342,82 s de travail,
784 243 728 octets d’archives. Aucun calcul d’apprentissage ni GPU.
Résultats et 216 NPZ committés 3eb5b95 ; reçus de lancement 21922f6.
Sources bd0b314, préflight 4034174, 26 tests synthétiques PASS.
Ne modifier aucune des 81 sources de cette collecte. Aucun rejeu.

Revue indépendante 910ff8c : 81 sources et 276 preuves exactes,
216 archives / 1 512 membres ZIP, 60 724 événements vérifiés sans tableaux
NPZ décodés. survey_review_v0.json SHA3211472c93d6d910b477540fe8384cfdc49bfd112c875a9a3c7b736ee980924f.
Résultat survey_v0.json SHA877dc75eb44141d89395f009f1325cffd87b7ea13f3c3ea247c7d5f6cd45b61d.
104 paires contrastées : Pong18, Breakout18, Freeway24, Seaquest0,
MsPacman12, Asteroids32. Les douze fenêtres sont des données de développement.
Les zéros de Seaquest ne décrivent que ses deux fenêtres observées.
Comptages finaux brut/poolé égaux, PAS une preuve de masques identiques.

Figure f527aea : craft_v6_out/v70_action_signal_survey/action_signal_v0.png,
source 8bbcfbe, métadonnées action_signal_figure_v0.json. Générée une fois
à partir des JSON, vue et vérifiée. Ouverture dans Codex renvoyée queued.

## ARC 4 : premier fit Atari terminé, résultat négatif revu

Le dérivé des cibles est complet : 216 archives, 12 ancres, 5,0051 s CPU,
zéro nouvelle émulation. Résultat a7ab5d2, revue 0a99bab ; dossier
craft_v6_out/v70_response_envelopes/. Counts : 144/296, 344/184, 127/127,
0/0, 109/184, 78/54. Somme 1 647 pixels dans les douze masques distincts.
Résultat SHA d3c662ac1c269ff723add3e4417689a9dc57e7aa1745e9a700a7940bf78b6659,
NPZ SHA3221b9bae8d2c4f3af771f7af6ee08a66733d2e5c6744c917be601d88c31e03f,
revue SHA d1211d166c2af7a402587c3225f981d273392be9b08277c1da10419d12d825a2.
85 sources et 226 preuves vérifiées ; le réviseur a décodé seulement le dérivé.

Le premier lancement de ce constructeur avait été refusé AVANT tentative,
à cause de Git 2.24.1 qui renvoie littéralement --show-object-format. Observation
préservée, préflight v0 intact. Correction 7ae4725 : plomberie hash-object sur
entrée vide vérifiée dans proofs(). Préflight v1, 13 tests PASS, puis une seule
construction réelle réussie. Aucune archive/ancienne preuve supprimée.

Fit : scripts/fit_response_envelope_v70.py, modèle response_envelope_v70.py,
tests et docs ARC4_RESPONSE_ENVELOPE_FIT_* ; sources 842fd9b puis 4138fb8.
Préserver les 90 sources ayant produit ces résultats. 19 tests PASS en 7,16 s.
Préflight cab625d SHA dc49248084aad273fe3a988af15715850b80f05a16aeb2347e98792c61cf37a2.
Le vrai configure_runtime(); proofs() a passé 90 sources et 3 preuves, sans
modèle ni tableaux. Les trois défauts du lanceur relevés ont été corrigés avant
ce départ : step partiel incertain, poids perdus sur interruption/panne du
journal, mesure trop tôt de la fin du budget.

Lancement WMI caché 10 septembre 02:37:39Z (04:37 Paris), PID17684, hors Job
Object vérifié par la console. Intention fsync avant WMI et reçu dispatch retour0
committés dans v70_response_envelope_fit/launch_v0/. Processus 17684 maintenant
absent : exécution complète en 738,146 s, 256 updates par bras, 512 au total,
compte exact sans update incertaine ni erreur. Artefacts, poids et console
committés 3473f13 puis 963945e. Aucun autre fit lancé.

Contrat fixe : deux images RGB passées, FCN 10 177 paramètres (16 canaux,
dilatations1/2/4/8/16, padding répliqué), graine764, visuel contre entrée zéro,
poids initiaux identiques, 256 updates par bras alternées, lot complet12.
Adam3e-4, BCE équilibrée par ancre, témoins initial et sortie zéro. CPU1,
CUDA vide, 1 200 s de travail dont réserve10s pour clôture. Le payload complet,
son fsync/hash et la clôture du journal sont dans ce temps ; seul le petit reçu
final fit_v0.json et la console sont hors mesure. Aucun retry ou prolongement.
Les paramètres/métriques ne doivent pas changer après lecture des résultats.

Résultat final fit_v0.json SHA9472d1153ea590b50fb3bd0a8f37baf5a989392604ade6bde60501a5155bad4a.
Critère NON atteint : IoU moyen 0,04739 contre minimum 0,25 ; avance 0,04249
contre minimum 0,10 ; faux positifs Seaquest 0,5655 % et 0,1607 %, chacun
au-dessus du maximum 0,1 %. Les trois clauses échouent. AP moyenne 0,15666.
Il s’agit d’un résultat scientifique négatif, pas d’une erreur du lanceur.

Critère : IoU visuel moyen par jeu >=0,25, avance >=0,10 sur meilleur des
3témoins, faux positifs <=0,1% pour CHAQUE ancre vide, 256+256updates complets.
AP groupant les ex aequo publiée sans seuil supplémentaire. Douze situations
DE DÉVELOPPEMENT, aucune validation indépendante ni objet physique démontré.
Même un succès ne signifie ni jouer, ni généraliser, ni transférer plus vite.
Les 1 280 updates antérieures étaient SYNTHÉTIQUES ; les 512 de ce premier
fit sur les images Atari se comptent séparément. Aucune policy de jeu entraînée.

Revue indépendante déposée : fit_review_v0.json, SHA
3df95a49c616d980c940b9ae20260a8e5c912560eecf07c6e0a8ad7f122d0467.
90 sources, 15 preuves et 2 049 événements vérifiés ; recalcul des 48 lignes
de métriques conforme à 1,11e-16 près, sans inférence. Validité instrumentale
PASS, résultat scientifique FAIL. Témoin constant uniforme 0,416665524,
proche de son optimum analytique 5/12 ; l’initial prédit tout positif.
À faire : committer cette revue puis générer et voir la
figure JSON-only préparée 6e5099d (PAS encore exécutée) :
scripts/figure_response_envelope_fit_v70.py attend fit_review_v0.json de kind
response_envelope_fit_independent_result_review. Aucun tableau NPZ ouvert par
cette figure. Voir le PNG après génération, committer, poursuivre au jalon suivant.

Hypothèse source : ARC4_RESPONSE_ENVELOPE_DRAFT.md. La cible d’écarts futurs
est distincte de C/m/ME, sans tracker ni mouvement préalable. Pas de bouton,
jeu, coordonnée ou oracle en entrée. L’ancien candidat de table de codes locaux
contredirait ARC4_VISION §2.6 ; ne pas l’implémenter. Pas de modèle conditionné
par bouton, ni contrôle/policy/ME construit par ce fit.
Validation conditionnelle préparée AVANT résultat du fit :
docs/ARC4_RESPONSE_ENVELOPE_VALIDATION_DRAFT.md. Nouvelles graines19490/19491,
seulement si fit complet + critère atteint + revue PASS. Cette condition a
ÉCHOUÉ : ne pas collecter cette validation. Aucun nouveau tableau collecté ;
aucun script de validation écrit. Douze ancres, critères et censures
prédéclarés ; 30k charges/jeu,900s collecte +300s dérivation +180s évaluationCPU.
Un doublon de passé du développement => global non concluant, sans remplacement.
Cette validation ne démontrerait pas le transfert vers un autre jeu.

Prochain diagnostic en discussion : chercher des passés locaux exactement
identiques associés à des cibles opposées, puis borner l’IoU et la BCE du FCN.
Attention au padding répliqué à chaque couche : un patch brut répliqué seul
ne suffit pas aux bords. Raffinement sous revue : joindre à la clé externe
les quatre distances aux bords écrêtées au rayon 31. Aucun code, protocole
ou calcul sur données réel de ce diagnostic à ce stade ; aucun nouveau fit.

## Limites antérieures préservées

Rectangle natif : 518,35 pas/s contre baseline580,79 ; gain non établi.
Résultats18bbeb6, revue13be4ae, 16 rapports identiques, 171 364 charges.
Source native_chain_v0.json SHA35742d06e6df89839527b2dbbc187aca7e2970778ecf0a9ede3aaa5a8bc76d4b.
Charges CPU concurrentes différentes : pas de régression causale établie.
Pas de nouveau benchmark. Diagnostic de profilage encore brouillon dans
ARC4_NATIVE_CPU_PROFILE_DRAFT.md ; ne pas rejouer les anciens profils.

C150 instances : zéro supervision, non résolu. Cinq fits SYNTHÉTIQUES,
1 280 mises à jour au total, tous négatifs au critère global. Fit Atari négatif séparé.
Gate0 non gelé, 12GPUh/0B10 et STOP inchangés. Jeux réservés Boxing,
Frostbite, Riverraid, Kangaroo jamais ouverts. RAM : évaluation externe
uniquement. ME/boutons/reader/drives/S/policy/ledger restent à construire.

## Environnement et discipline

SETUP.md lu. Main C:/Python314/python.exe3.14.3, NumPy2.4.4, Torch2.11+cu126.
.venv-arc4 : Python3.14.3, ALE0.12.1, sans Torch. CPU : CUDA_VISIBLE_DEVICES
vide, OMP/OPENBLAS/MKL=1. Pas d’installation ni de rebuild automatique.
Exclure .codex/, configurations privées, smokes/notches et sorties ARC3
actives. Pas de suppression ni de push ; commits ciblés, git commit -q.
Ne pas relire l’intégralité de Claude ou de la tâche. Français simple.

Automatisation unique suivi-ragnarok-arc-3, mise à jour pour cette reprise :
PID réels, silence sans changement utile, pas de doublon/retry automatique.
Tâche 01a082c5-c85d-7031-95e4-8d432972cb6e. Ne pas réveiller audit_gate_v62.
