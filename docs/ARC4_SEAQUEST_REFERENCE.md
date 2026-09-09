# Seaquest : référence visuelle causale

Protocole et exécutable `81a7fa4`, revus par Meitner avant la capture uniforme
17371 :8192images, six constructions froides,8960frames facturées. Le prérequis
de calibration brute commitée a été vérifié avant la première construction.
Cette séquence est distincte des collectes de calibration, dont l'usage pour
mettre à jour le fond de référence reste interdit.

L'extracteur reçoit uniquement RGB et conserve son histoire causale après les
morts. Le juge externe utilise la RAM, jamais l'extracteur. La mesure couvre
toutes les images et tous les objets visibles dans la région annoncée.

Résultat de développement, aucun gate : rappel mobile17607/18803=93,6393 %,
précision visible20598/81013=25,4255 %, donc60415prédictions non appariées.
Le rappel agrégé masque un joueur difficile :2661/3638=73,1446 % ; missiles
joueur1021/1219=83,7572 %, plongeurs6294/6299=99,9206 %, requins7631/7647=99,7908 %.
Sous-marins et missiles ennemis absents : dénominateurs nuls et rappel `null`.
Cette collecte uniforme ne calibre pas leur qualité de référence.

Débit local558,47images brutes/s pour décodeur, référence NumPy et scoreur,
hors émulateur, IPC et branches. ARC3 tournait simultanément. Ce chiffre n'est
pas le débit complet à16environnements. Versions, temps et empreintes conservés
dans `craft_v6_out/v70_seaquest_reference/raw_reference.json`.

Les clés RAM indiquent des emplacements d'allocation. La mobilité est un
changement de centre visible sur huit frames ; animation, clipping et réemploi
d'un emplacement peuvent l'influencer. Aucun apprentissage de l'agent ni
supervision gratuite de son futur parseur ne découle de ce diagnostic.
