# Seaquest : chaîne CPU complète

Protocole et raccord revus par Ohm, commit `2cb7b29` avant nouvelle émulation.
Les juges brut et temporel restent inchangés. L'adaptateur utilise un graphe
des rectangles originaux, avec la même connexité et le même ordre que le raster.
Égalité vérifiée sur 256 arrangements synthétiques et les 16 384 fenêtres
complètes de développement et de validation réservée.

Le scoreur compte chaque composant temporel une fois, même si deux composants
partagent les mêmes emplacements RAM. Un composant est mobile si au moins un
de ses membres a changé de centre entre t−8 et t. Tous les composants visibles
participent à l'appariement global unique. Ce rappel n'est pas directement
comparable au rappel des acteurs bruts : fusion, animation, occultation et
réemploi d'un emplacement influencent sa définition.

Raccord réel : pixels, boîtes de référence et cibles C identiques lorsque les
deux décodeurs RAM sont vidés ; alignement conservé au terminal, après les
restaurations et après reprise froide ; RPC oracle indisponibles et processus
fermés. Coût des cinq instances : 3 582 frames facturées. Reçus de géométrie et
de raccord committés `a025cb9` avant benchmark, avec 53 empreintes de sources.

Deux mesures de 16 × 2 048 décisions, RGB IPC complet, référence causale,
jugement à chaque frame réelle et branches toutes les 128 décisions :

| Référence | Travail | Décisions/s | Démarrage |
|---|---:|---:|---:|
| NumPy | 32,7732 s | 999,8425 | 4,3708 s |
| Native | 24,9839 s | 1 311,5635 | 3,8233 s |

Chaque mesure facture 174 380 frames : 130 868 de jeu ordinaire, 84 × 128 de
constructions froides, 2 040 de branches et 512 × 60 de restaurations.
NumPy est légèrement sous le minimum technique de 1 000 ; le natif le dépasse.
La cible de 2 000 reste non atteinte. ARC3 tournait simultanément : ces mesures
ne démontrent pas un gain contrôlé ou universel.

La qualité reste limitée : rappel des composants mobiles 82 656/88 404
(93,4980 %), précision visible 95 850/343 701 (27,8876 %). Le débit ne corrige
pas les nombreuses détections parasites. Aucun parseur appris, lecteur, pulsion
ou apprentissage du jeu n'est inclus ; ce résultat n'est pas un gate ARC4.

Revue indépendante Ohm close : les 16 rapports complets sont exactement
identiques, ainsi que les 53 empreintes de sources, les preuves et les comptes.
Le build et la DLL ont été vérifiés. Il s'agit de l'égalité des rapports
enregistrés, sans seconde comparaison de chaque image du benchmark.
Les rapports et le reçu `cpu_backend_equivalence.json` sont conservés dans
`craft_v6_out/v70_seaquest_pooled/`.
