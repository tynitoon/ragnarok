# Décodage externe des pastilles par lecture de huit pixels

9 septembre 2026. Candidat séparé `gather_visible_mspacman_v70.py`, sans aucune
modification des sources ou résultats gelés. Il concerne uniquement le juge
externe MsPacman ; il ne doit jamais alimenter les observations ou objectifs
de l'apprenant.

Le précédent candidat par image intégrale a été plus lent sur l'audit offline
complet ; son résultat négatif est conservé. Ce candidat lit directement les
huit pixels de chaque pastille nominale4×2 dans le masque d'occupation existant.
Une lecture NumPy groupée remplace les nombreuses petites réductions `any`.
Une pastille touchée conserve exactement l'ancien découpage. Les super-pastilles
clignotantes restent entièrement dans l'ancienne fonction générale.

Les tables de positions, l'ordre des labels, les listes de sprites, les supports
historiques d'occultation et les corrections des fantômes/yeux restent ceux du
décodage V2. Le test AST vérifie que le corps source ne diffère que par cet appel.
Les tests synthétiques couvrent les256masques4×2 sur chaque grille et100cas variés.

Aucun temps d'exécution n'a encore été mesuré, aucune archive de calibration
n'a été réanalysée avec ce candidat, aucun natif n'est lancé. Une éventuelle
équivalence exhaustive et sa mesure offline devront avoir leur protocole
prospectif committé avant exécution, puis une revue indépendante. Aucune promesse
de gain ni substitution à la chaîne cache actuellement scellée n'en découle.
