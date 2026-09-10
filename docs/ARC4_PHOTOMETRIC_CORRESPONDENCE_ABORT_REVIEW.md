# ARC4 — revue indépendante de l'arrêt photométrique

10 septembre 2026. Revue de `audit_motion_bootstrap`, limitée aux sources,
reçus, répertoire et empreintes. **Arrêt technique confirmé ; aucun résultat
scientifique du candidat.** Aucune relance autorisée par cette revue.

## Stade atteint et comptes

La tentative commence à `2026-09-10T08:25:55.379804+00:00`. Le reçu de
génération annonce son achèvement à `08:25:57.191838+00:00`, puis le processus
`fit` échoue à la ligne 93 de `scripts/photometric_worker_v70.py` :
`ValueError: CUDA must be unavailable to this CPU experiment`.
Le parent conserve `execution_status="failed"`, le code de sortie enfant 1
et `elapsed_seconds=7.4001315`, relevé avant l'écriture du reçu d'échec.
Cette valeur n'est pas une attestation positive de clôture après fsync.

La ligne 93 suit les imports NumPy/Torch/module du candidat, la configuration
des threads et l'appel `torch.cuda.is_available()`. Elle précède la lecture du
protocole dans `fit`, la définition des compteurs et l'ouverture de son journal.
Le premier `rgb_only` est ligne 197, `manual_seed(782)` ligne 198 et la
construction du réseau ligne 199. L'import du module définit la classe ; il ne
construit pas d'encodeur.

Par le traceback et cet ordre de contrôle, **dans cette tentative de
production** : zéro décodage RGB par le learner, zéro initialisation 782,
zéro modèle candidat, forward, objectif, backward, gradient ou mise à jour ;
zéro association initiale/finale/témoin et zéro appel du scorer. Ce sont des
comptes déduits du stade d'arrêt, pas des compteurs retournés par le worker.
`fit_worker_v0.json`, `updates_v0.jsonl`, tous les checkpoints attendus,
`fit_v0_completion.json` et les artefacts de jugement/score sont absents.
Les fixtures antérieures constituent un autre périmètre : leur backward de
vérification ne doit pas être effacé par ce bilan nul de production.

L'appel de disponibilité CUDA a retourné vrai. Ces preuves ne déterminent pas
la cause du masquage inefficace sous Windows ni l'activité interne du pilote.
Elles n'indiquent aucun calcul du modèle sur GPU. Aucun diagnostic CUDA n'a été
exécuté par le réviseur.

## Conservation et provenance

Les marqueurs de sauvegarde et le manifeste concordent : 128 clips
d'apprentissage et 1 280 clips d'évaluation, donc
`(128 + 1280) × 3 = 4 224` nouvelles images de la famille ; huit clips de
contrôle, soit 24 vues sélectionnées, et zéro frame émulée annoncée.
Les archives RGB et les métadonnées séparées du juge sont conservées.
Les nombres sont vérifiés contre les reçus et le code, **sans décoder ni
recompter le contenu des tableaux** et sans lire les labels détaillés.

Les 27 sources ont leurs SHA identiques dans préflight, intention et reçu,
correspondent aux fichiers présents, sont suivies par Git et sans différence
avec HEAD. Les 30 fichiers de l'inventaire ont chacun la taille et le SHA
annoncés ; les sept liens du manifeste de génération concordent aussi.
L'inventaire représente 3 458 971 octets ; avec le reçu final de 8 645 octets,
le répertoire contient 3 467 616 octets à la revue, sous 64 000 000.
Le préflight, son runtime et la clôture positive des fixtures sont cohérents.
La vérification finale du chemin de succès du fit n'a pas été atteinte ; les
contrôles d'empreintes ci-dessus sont ceux de cette revue après l'arrêt.

## Preuves utilisées

Chemins du tableau relatifs à `craft_v6_out/v70_photometric_correspondence/`.
Les SHA des 30 fichiers vérifiés sont également conservés dans l'inventaire du
reçu `fit_v0.json`, dont l'empreinte fixe cet inventaire.

| Preuve | SHA-256 |
|---|---|
| `fit_v0.json` | `0dd96f31d2730fdd514d83cb204f6af2d40c79b411ad8a2fc69a3d646729f9f1` |
| `fit_v0_attempt.json` | `08d757c81e8286cbc0cbfac9ff5f18305e72b4d5846ad08aaf729dc78996cf41` |
| `fit_v0_output.txt` | `e8d45f13ef05558dbe7583e6a5a1a7613a639171b785166f296d191ff2b50876` |
| `generation_v0.json` | `fe255e8871ff0de2aebc0ebb66ce6c50a2745ec6b3ab8d363991c8259d81e6bf` |
| `generation_train_saved_v0.json` | `2bcb179844df12be68294b7a7779db52f826c91f0bab9197b71b3012f8cc82f1` |
| `generation_eval_saved_v0.json` | `75d6c9605fb56388f9edde4e64e80d1320b557f22ff34149a9683397b6adcabe` |
| `preflight_v0.json` | `fc74ab2e1392b0db901d90295bc312ca82df62514ef0ae814f363068640a40ee` |
| `check_v0.json` | `d1698c6fa7ec8d8dfa89b4dbc3b7a52eaa925640a348103bda874351b5bd4465` |
| `check_v0_completion.json` | `c1294cd50e988672c9c52a3dd2c754ae80ce84b0c216823ef83bc558da562bf8` |
| `train_rgb.npz` (octets seulement) | `6879781ed84b7ba00434ef762528b824868d64b7e643cca2766f9a78b8fd656a` |
| `eval_rgb.npz` (octets seulement) | `61334ce105e0ea773bf5d32b10048ff6213d67b1505469d5b75a76ebcbf93701` |

Sources décisives relues, chemins relatifs au dépôt :

| Source | SHA-256 |
|---|---|
| `scripts/photometric_worker_v70.py` | `b183c7d340b243b1dbd038b74780691635de4bdd8256ee9b1df39da47665db9d` |
| `scripts/photometric_correspondence_v70.py` | `777fdd4c646fb2c4a2b656e8ce2b01d198c67c8a3ff4bfa8b10afa88bea07272` |
| `scripts/photometric_common_v70.py` | `a498cf44701e9dcecbebb80e29894bb800527852b66f68e5d942c3ad4a521415` |
| `ragnarok/learning/photometric_correspondence_v70.py` | `1c37c933339e0de21557463349969658ef21e769775d50a9ff774109a84e8062` |
| `docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_PROTOCOL.json` | `e623403aa384bc11c90d160439323548b4ba8bf226c193948749a57c8c4d17db` |
| `docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_DRAFT.md` | `f3d5d34b958b40a0fc819a71bdc9296097fb0d21f19d4df174ac0e9d2666576b` |

## Portée et STOP

Cet arrêt ne constitue ni un succès ni un échec scientifique de l'association
apprise : aucun apprentissage ni score n'a eu lieu. Il ne produit aucune preuve
nouvelle sur m/C, ME ou le transfert. Le contrat impose : « Toute erreur,
dépassement ou non-finitude arrête sans recommencer ». La tentative et ses
artefacts restent donc conservés ; aucun juge ni nouvelle tentative n'est
déclenché par cette revue. Une éventuelle proposition technique distincte doit
expliciter ce changement de statut, sans effacer cet arrêt.

Coût de cette revue : lectures de texte/JSON, inventaire et SHA d'octets
existants seulement ; zéro import Python, test, tableau décodé, génération,
émulation, modèle ou mise à jour. Seul le présent document est créé, sans commit.
