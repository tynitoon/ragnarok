# Ms. Pac-Man : préparation, aucune capture de calibration ouverte

Notes de lecture et préparation du protocole de développement. Le périmètre
prospectif est maintenant fixé dans `ARC4_MSPACMAN_RAW_PROTOCOL.json` ; aucune
capture de calibration n'est encore ouverte. Les offsets restent des hypothèses.

Sources primaires MIT à révision fixe :
[RAM OCAtari](https://raw.githubusercontent.com/k4ntz/OC_Atari/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/ram/mspacman.py),
[vision OCAtari](https://raw.githubusercontent.com/k4ntz/OC_Atari/467accd5f3da379d83705b008094ffb787eec8a1/ocatari/vision/mspacman.py).
Conserver l'attribution et `docs/licenses/OCAtari-MIT.txt` pour toute adaptation.

Le périmètre fixé avant capture inclut joueur, tous les fantômes, fruit,
grandes pastilles et petites pastilles, soit cinq catégories séparées. Les comptages
maximaux de la source portent eux-mêmes une réserve de validation. Ne pas
assimiler une boîte nominale à son support visible ni omettre un type après
avoir constaté qu'il est difficile ou absent.

Points de départ documentaires : joueur RAM10−13/RAM16+1, fantômes RAM6..9−13
et RAM12..15+1, fruit RAM11−13/RAM17+1. Fantômes conditionnés par RAM47 et RAM19,
fruit par les deux coordonnées positives. Dimensions nominales communes9×10 ;
elles devront être confrontées indépendamment au RGB. RAM56 décrit l'orientation.

Les fantômes vulnérables partagent une couleur : le checker devra conserver
toutes les instances, y compris les contacts ambigus, sans déduire une identité
du seul ordre des composantes. HUD/vies/fruit décoratif doivent avoir un
périmètre explicite. Les petites pastilles utilisent deux grilles et un codage
de bits ; chaque cellule est une instance, avec mesure séparée pour empêcher
leur nombre de masquer la qualité du joueur.

Éviter de copier le cycle mutable des objets de la source : certains chemins
créent un objet dans la liste puis renseignent encore l'ancienne variable
locale. Un décodeur fonctionnel par frame rend cette classe de décalage inutile.
La source constitue une hypothèse à vérifier, pas l'instrument déjà validé.

Aucune donnée de confirmation Atari, NES ou autre génération n'est ouverte.
