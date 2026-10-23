# v84 — notes de construction (choix d'implémentation ; aucun critère, seuil, bras ou graine changé)

Le protocole qui fait foi est `docs/ARC4_V84_PROTOCOL_EN.md`. Ce fichier note les précisions
d'implémentation prises là où le texte laisse un choix, et tout écart, avec sa date.

1. **C₋g (A.5).** « Covariance de x̃ pondérée également par jeu » : moyenne sur les 40 jeux sources des
   covariances de x̃ autour de la moyenne pondérée également des moyennes par jeu,
   C = D⁻¹[(1/G)Σ_h M_h − a aᵀ]D⁻¹ avec M_h le second moment brut du jeu h, a = (1/G)Σ_h μ_h et
   D = diag(max(sd, 0,05)) (sd poolé sur toutes les lignes sources).
2. **T7 (test gaussien).** La première version du générateur (2 000 lignes, 20 dimensions) était trop
   facile : tous les bras franchissaient chaque niveau au premier point de mesure (rapport exactement 1),
   ce qui ne teste rien. Générateur utilisé : 6 jeux, flux de 3 000 lignes, 60 dimensions, poids de
   norme 0,8, biais −1,5, sous-espace commun de dimension 3 ; κ = 0,25 pour les deux bras. Résultat :
   commun 2,15 (> 1,5), indépendant 0,95 (≤ 1,25). C'est un test de justesse du code, pas un réglage du
   protocole ; écart au texte (« 2k-row streams, 20 dimensions ») noté ici (23 octobre).
3. **T8 (contrôle planté).** La base de plantation x̂ est la géométrie A.5 des 12 jeux réunis (pour que
   « sous-espace commun » ait un sens unique) ; le pipeline, lui, recalcule sa géométrie en laissant un
   jeu de côté, comme G0. Les flux plafonnés à 6 000 lignes sont tirés parmi les lignes stockées (déjà
   plafonnées à 24 000) avec `default_rng([9701, game_index, seed])`. Seule la tête « BON » est plantée.
4. **P_shift (rapport seulement).** Décalage circulaire par flux (lignes non censurées), au moins 10
   déroulés de lignes, `default_rng([9707, game_index, seed, head_index])` ; un flux trop court exclut la
   tête du bras P_shift.
5. **K\*Beven (rapport seulement).** Lignes des blocs pairs de B, privées des lignes dont le hachage
   figure dans l'ensemble de test, tête ajustée comme K\*_A (ρ par 2 plis blocs pairs/impairs).
6. **Certificat de préfixe sur les flux réels (vi).** Vérifié courbe par courbe : tout bras à moyenne
   nulle vaut exactement 0,5 à chaque point de mesure antérieur à la première ligne qui complète les deux
   classes.
