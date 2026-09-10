# Proposition de reprise technique — autorisation de Jérémie requise

La tentative du 10 septembre 2026 à 10:25 Paris est close et conservée,
sans apprentissage, sur la garde d’environnement CPU. Le contrat gelé impose :
« Toute erreur, dépassement ou non-finitude arrête sans recommencer ».
La présente proposition **ne lève pas ce STOP**. Le lanceur de reprise ne doit
être exécuté qu’après accord explicite de Jérémie à cette exception technique.

## Correction vérifiée et périmètre

Le diagnostic séparé a comparé deux processus frais : chaîne vide → CUDA encore
disponible ; `CUDA_VISIBLE_DEVICES=-1` → CUDA indisponible. Il n’a construit aucun
tenseur, modèle ou optimiseur. Durée 6,6171842 s. Le réglage -1 est documenté
par [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/cuda_environment_variables.html).
La cause interne du comportement de la chaîne vide n’est pas déduite de ce test.

La reprise proposée change uniquement le masquage GPU et la destination des
reçus. Les 27 sources de la tentative close restent intactes. Le wrapper importe
leur fonction de fit inchangée et redirige ses écritures vers un autre dossier.
Les quatre archives RGB, les fichiers du juge et le manifeste sont copiés par
octets vérifiés depuis l’arrêt ; **aucune scène n’est régénérée**. Aucun label
du juge n’est lu par le processus d’apprentissage ou les contrôles de reprise.

L’architecture, l’initialisation 782, les 512 mises à jour, les lots, les seuils,
les neuf critères, les cinq strates et le juge restent ceux déjà fixés. Aucun
résultat de modèle de cette famille n’a été observé. Les deux états et le témoin
RGB seront toujours comparés après le fit, puis scellés avant le juge.

## Exception demandée et limites

Une seule reprise opérationnelle, **280 secondes murales** sur un CPU, imports,
copies, fit, inférences et clôture compris ; réserve 15 s. C’est une dérogation
explicite au lancement unique et à la fenêtre murale continue d’origine.
Les pauses de diagnostic humain ne deviennent pas du temps de calcul disponible.
La convention proposée débite 8 s pour le lancement arrêté et 7 s pour le
diagnostic : 8+7+280=295 s de calcul actif comptabilisé. Le reçu d’arrêt relève
7,4001315 s avant son écriture ; il ne possède pas d’attestation positive après
clôture. Ce débit est donc une **convention explicite de l’exception demandée**,
pas une nouvelle preuve autonome d’un plafond réel cumulé. Les fixtures et
revues en lecture seule gardent leurs périmètres distincts déjà documentés.

Le plafond de 64 MB s’applique à la somme des dossiers d’origine, de diagnostic
et de reprise. Les fragments et fichiers pending restent conservés. Une erreur
de cette reprise termine la tentative sans boucle de relance. Les attestations
positives et la frontière de publication de l’amendement 2 restent exigées.

Le juge garde son budget séparé de 60 s et ne pourra être exécuté qu’après
clôture complète et revue indépendante de cette reprise. Aucun raccord m/C,
ME, Atari, gate 0 ou conclusion de transfert n’est autorisé automatiquement.

## Action concrète préparée

`scripts/photometric_technical_recovery_v70.py preflight` ne vérifie que des
octets, sources et reçus, sans modèle ni tableau décodé. Le mode `run` exige
le flag explicite `--owner-approved-technical-recovery`, à utiliser seulement
après la réponse de Jérémie. Dossier distinct :
`craft_v6_out/v70_photometric_technical_recovery/`.

Le premier arrêt, son diagnostic, les tests et la revue indépendante ne sont
ni remplacés ni requalifiés comme un résultat scientifique négatif.
