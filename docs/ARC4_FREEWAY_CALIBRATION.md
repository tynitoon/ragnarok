# Freeway — calibration brute CPU

Le protocole `ARC4_FREEWAY_CALIBRATION_PROTOCOL.json` est établi avant la capture
17211. Validation 17212 fermée. Douze objets : deux poulets, dix voitures par
voie. Le deuxième poulet stationnaire participe à la présence et à la précision.
Il ne compte comme mobile que si ses positions changent dans l'historique.

Les coordonnées RAM et les couleurs viennent des modules MIT
[RAM Freeway](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/ram/freeway.py)
et [vision Freeway](https://github.com/k4ntz/OC_Atari/blob/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/vision/freeway.py)
d'OCAtari, commit fixé. Le checker indépendant cherche les pixels dans des zones
fixes, sans appeler le décodeur RAM ni recevoir ses boîtes. Aucun label oracle
n'entre dans la référence ou l'agent.

La baseline utilise les dimensions du port, clippe le terrain et conserve les
éventuelles erreurs de bouclage, d'occlusion ou d'animation. Les seuils de 99 %
de présence et de boîtes à un pixel seront appliqués à chaque type. Toutes les
frames restent incluses. Un résultat négatif sera conservé avant correction.
Pas de gate ni d'entraînement. L'absence de vies ALE dans ce jeu interdit de
présenter sa calibration comme validation d'une mesure de mort.
