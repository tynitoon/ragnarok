# ARC 4 — accélération prospective du juge externe

Le profil offline 17471 donne, avec surcoût du profileur : décodage 13,148 s,
groupement 7,319 s, historique de mouvement 4,439 s, appariement/scoring 3,139 s,
référence RGB native 1,531 s. Il comprend 8192 images et 2051 regroupements
synthétiques, dont 329 mélangent plusieurs actions archivées. Ce diagnostic ne
mesure ni émulation, ni IPC, ni branches, ni débit de la chaîne native.

Deux nouveaux adaptateurs conservent les anciennes sources et leurs résultats :
`vector_visible_mspacman_v70.py` utilise une image intégrale pour repérer en lot
les pastilles sans occlusion ; toute pastille intersectée passe par le même
découpage que précédemment. Le reste du décodeur conserve les supports, l'ordre,
les identifiants et les géométries V2. Aucun cache de trajectoire n'est introduit
dans ce décodeur, aucune RAM n'est modifiée.

`cached_visible_groups_v70.py` conserve uniquement la dernière entrée par type
externe et son résultat géométrique exact. La clé contient les objets sources
complets, y compris leurs identifiants : des rectangles identiques avec des
appartenances différentes ne partagent pas une réponse. Les entrées sont validées
avant la recherche ; les tableaux raster ne sont pas conservés. Une erreur ne
remplace pas le cache antérieur. Les types vides donnent directement une liste
vide. L'ordre de scan et les unions de membres restent ceux du juge précédent.

Les tests vérifient les 256 masques d'occlusion d'une pastille4×2, 100 populations
de rectangles et les changements de géométrie/appartenance de deux jeux. Cela
ne remplace pas l'équivalence exhaustive sur les archives ni une mesure native.
Ces adaptateurs restent exclusivement des instruments d'évaluation, sans entrée
dans l'apprenant. Aucun gain de débit, nouveau PASS de calibration ou gate n'est
déduit de leur existence. Un protocole et une revue seront nécessaires avant une
nouvelle chaîne native ; les 115 sources du benchmark original sont conservées.
