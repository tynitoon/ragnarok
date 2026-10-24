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
7. **T8, premier passage (23 octobre, code 07a71bc, avant la revue T9) — compté comme premier échec
   positif.** Positif : R(K\*A:F) = 1,62, R(X:C\*) = 1,07 (échec) ; nul : R(X:F) = 0,60, R(X:C\*) = 0,60
   (réussi). Gardé dans `craft_v6_out/v84/g0/attempts/planted_attempt1_07a71bc.json`. Ce résultat a été
   lu avant la revue ; par prudence il compte comme le premier échec positif d'A.15 T8 : il reste donc
   UNE modification de recette, puis un seul nouveau passage ; un second échec positif donne
   RECIPE-INSENSITIVE. Diagnostic (étiquettes plantées seulement) : les têtes sources retrouvent mal la
   direction plantée (cosinus médian 0,30 ; la direction du jeu tenu à l'écart n'est qu'à 40-50 % dans
   l'espace des têtes), le bruit d'estimation se loge dans les directions de très faible variance.
8. **Modification de recette unique (A.15 T8, déclarée avant le second passage).** Seuil d'espace nul de
   A.5 relevé de 1e-5 à **1e-2 × λ_max** (bouton autorisé « null-space threshold ») : sur la géométrie
   de 40 jeux, 88 directions gardées (85 % de la variance de x̃) au lieu de 342. Choix par raisonnement
   (les directions écartées portent ~15 % de la variance et l'essentiel du bruit des têtes), sans essai
   de valeurs sur T8. Aucun autre bouton n'est changé. S'applique à G0 entier (géométrie de tous les bras).
9. **Corrections de la revue T9 (`wf_d1fa216a-b76`), qui ne sont pas des modifications de recette :**
   K\*A ajusté avec une seule ordonnée sur tout le flux A (ρ par 2 plis pairs/impairs) comme le dit A.7 ;
   K\*Beven n'était jamais calculé (plis tirés des blocs pairs : (bloc // 2) % 2) ; contrôles (vi) rendus
   capables d'échouer (enregistrement machine des tests, `planted.json` réussi et inchangé, empreintes
   des entrées de chaque étape, arbre de code propre ancré au worktree, fichiers du magasin égaux à
   `store.json`) ; certificat de préfixe aussi sur F ; part N(0,25) avec K\*A ; profil R(q) rapporté ;
   tests d'agrégation ajoutés ; T7 ramené aux tailles figées (6 jeux, 2 000 lignes, 20 dimensions) avec
   la règle F seul ci-dessous, l'ancien générateur gardé comme test T7b.
10. **Règle de choix du générateur T7 aux tailles figées (déclarée avant de lancer X).** Grille
    wn ∈ {0,2 ; 0,3 ; 0,4 ; 0,5}, biais ∈ {−2,5 ; −2,0 ; −1,5} ; on prend le plus grand wn (puis le plus
    grand biais) pour lequel F seul (κ = 0,25) franchit N(0,25) après c(n₀) sur au moins 4 des 6 jeux,
    pour les graines 11 et 12 ; puis T7 tourne une fois. Si aucun réglage ne convient, on garde le test
    3 000/60 au titre « corrigé et noté ».
11. **T7, résultat de la règle F seul (23 octobre).** Aux tailles figées (6 jeux, 2 000 lignes,
    20 dimensions), nombre de jeux où F franchit N(0,25) après c(n₀), graines 11 (commun) / 12
    (indépendant) : wn 0,5 : (0,1) (1,1) (2,1) ; wn 0,4 : (0,1) (1,1) (3,0) ; wn 0,3 : (3,2) (1,1) (4,1) ;
    wn 0,2 : (4,3) (2,3) (5,1), pour biais −1,5 / −2,0 / −2,5. Aucun réglage n'atteint 4 et 4 : T7 garde le
    générateur 3 000/60 (note 2) au titre « corrigé et noté ».
12. **G0-S (préparé le 23 octobre pendant G1b, avant toute donnée G0-S).** Implémentation
    `scripts/g0s_v84.py` : flux d'expansions des sorties G1b (44 variables du sens de la recherche, ADV
    comme unique tête, horloge = images facturées à la fin de l'expansion moins l'entrée, blocs de 50
    expansions, « bouton » = processus pour le taux de base) ; témoin T_clock = têtes sources ajustées sur
    les 7 variables de contexte du run seulement (colonnes 19-25), les autres mises à 0 donc écartées par
    le plancher d'écart-type ; C\* = meilleur de {F, P, U, T_clock} ; règle de bord de κ étendue à
    T_clock ; seuil d'espace nul tel qu'amendé (1e-2).
13. **T8-S (contrôle planté de G0-S), déclaré avant toute donnée G0-S.** Même générateur que T8 (sous-
    espace commun de dimension 5, écart-type des logits 1,5, taux de base 0,15 ; nul : directions
    indépendantes), 12 jeux tirés avec 9708, κ sur 4, métrique sur 8 ; mêmes seuils de réussite. S'il
    échoue, G0-S = RECIPE-INSENSITIVE (voie S fermée, v84 s'arrête à la faisabilité), sans courbe réelle.
14. **Observation de méthode (synthétique seulement, 23 octobre).** Sur un jeu de données synthétique
    (sous-espace commun de dimension 3 dans 44 variables), l'horloge donne R(K\*A:F) = 1,59 et
    R(X:F) = 1,48 ; dans les deux passages de T8 sur la géométrie réelle, K\*A n'atteignait que 1,62 et
    1,44. Le plafond pratique de cette mesure semble proche de 1,5× même avec un a priori presque parfait :
    l'apprenant F atteint vite les premiers niveaux. Le seuil figé (ii) « K\* ≥ 2,0 » risque donc d'être
    hors de portée par construction. Rien n'est changé (seuils figés) ; leçon pour la suite : simuler le
    plafond atteignable d'un instrument avant d'en figer les seuils.
15. **Vérificateur de rejeu, G1b (24 octobre, après la première notation).** 2 des 200 vérifications
    échouaient, toutes deux au palier le plus profond de Q\*bert (12 225 rejoué 875 ; 12 275 rejoué 5 200).
    Diagnostic (run reproduit à l'identique jusqu'au palier, mémoire comparée à chaque sauvegarde) : la
    divergence naît dès le premier déroulé depuis la racine. V2 ne sonde pas à la racine : son premier
    déroulé restaure la racine juste après la remise à zéro, dans le contexte ALE « post-reset » hors de
    l'état sauvé (note 21 de v83) ; le vérificateur jouait toujours un NOOP avant de restaurer (contexte
    « après un pas »). Rejoués sans ce NOOP, les deux paliers retombent exactement (12 225 et 12 275).
    Correction côté évaluateur seulement : un palier est confirmé si le rejeu exact tombe juste dans l'un
    des deux contextes, et le contexte utilisé est noté. Première notation gardée
    (`room_verdict_v84_first_pass.json`, « Interruption ») ; verdict recalculé sur les mêmes fichiers et
    étiqueté « re-scored » (§3.3, point 5). Les comptes ne changent pas (22/35, se_proj 0,142).
