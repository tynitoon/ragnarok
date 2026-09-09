# Référence RGB — calcul compilé optionnel

Le backend `native_reference_v70.py` conserve l'algorithme causal de
`scripts/ref_extract_v70.py`. Il regroupe en C le packing RGB 24 bits et, lorsque
les couleurs sont déjà connues, l'incrément d'un compteur par pixel et la mise
à jour du mode par comparaison stricte. Les égalités conservent l'ancien mode.
Palette, limites, croissance, composants et sorties restent ceux de NumPy.

La première passe C valide toutes les couleurs et tous les IDs avant mutation.
Une couleur inconnue renvoie au chemin NumPy original, qui effectue une seule
mise à jour ou refuse la capacité dépassée. Compteurs uint32 limités à un million
d'observations ; calcul des indices en 64 bits. Chaque worker possède ses tableaux
et traite les appels séquentiellement. Aucun jeu, RAM, label ou reset oracle
n'entre dans ce calcul.

Construction explicite Windows x64 avec MSVC déjà installé :
`scripts/build_reference_native_v70.py`. Aucun remplacement du runtime ARC 3.
Le reçu `craft_v6_out/v70_native_reference/build_v0/build.json` conserve les
empreintes source, DLL, compilateur et linker ainsi que la commande. DLL sans
point d'entrée ni import, deux exports seulement, contrôlés dans la revue
indépendante Meitner. Le loader vérifie source et DLL avant utilisation ; aucune
compilation ou mise à jour implicite n'a lieu dans les workers.

Les tests comparent chaque état interne et les composants au chemin NumPy :
couleurs nouvelles, palette 4096, entrées non contiguës, égalités, refus atomique,
requêtes de branche sans changement du fond. La comparaison dans le vrai moteur
Pong/Breakout comprend reprise, images, boîtes, scores, cibles C et comptes.
18 tests natif/référence passent ; revue close sans défaut matériel.

## Chaîne Breakout complète

16 workers × 2048 décisions, répétition quatre, moteur froid, IPC RGB,
référence, juge à chaque frame et branches comprises, pendant ARC 3 :

| Implémentation | Décisions/s |
|---|---:|
| Géométrie temporelle raster, référence NumPy | 642,61 |
| Géométrie par intervalles équivalents, NumPy | 803,66 |
| Intervalles, calcul RGB compilé | 1005,41 |

Les **16 rapports sont strictement identiques** entre les trois variantes :
scores, comptes, cibles C. Le total est 186450 frames facturées. Le dernier
calcul dure 32,5916 s, démarrage séparé 3,4988 s. Reçus d'équivalence et sorties
complets dans `craft_v6_out/v70_breakout_pooled/`.

Ces exécutions successives ont subi une charge ARC 3 variable : les différences
de temps sont des diagnostics, pas une mesure contrôlée d'un facteur de gain.
Le minimum technique provisoire 1000 est juste dépassé ; la cible 2000 du
brouillon reste non atteinte. Aucun gate ni apprentissage n'en découle.
Le backend reste optionnel (`--reference-backend native --native-build ...`).
