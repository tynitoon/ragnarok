# v84 — notes de construction (choix d'implémentation et écarts datés ; aucun seuil, bras ou graine changé ; la note 16 modifie la méthode de rejeu du §7.2)

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
16. **Écart au texte du §7.2 (rejeu), 24 octobre, au titre du §3.3 point 5.** Déclencheur : première
    notation G1b = Interruption (2/200 vérifications, Q\*bert M7). Le §7.2 figé dit « rejeu de la
    note 21 (un NOOP avant de restaurer) ; un écart est une Interruption ». Depuis ac8b126, un palier
    est confirmé si le rejeu exact tombe juste dans l'un des deux contextes ALE (après un pas, ou
    post-reset) ; le contexte est noté. Cette règle vaut pour toute vérification de rejeu ultérieure.
    Effet : le re-scoring fait passer G1b d'Interruption (pas de verdict) à PROCEED, comptes
    identiques. Lecture exacte : 198/200 vérifications tombent juste avec le rejeu figé ; les 2
    échecs tombent juste seulement en post-reset (12 225, 12 275) ; 4 des 198 (Q\*bert M1/M2, deux
    graines) sont sur des lignées issues du même premier déroulé post-reset. Le vérificateur confirme
    que les scores enregistrés sont reproductibles, pas que chaque rejeu a pris le contexte de sa
    lignée. Robustesse (calcul à la main, non officiel) : sans Q\*bert, 21/35 jeux de place,
    se_proj ≈ 0,15, 20/35 « grimpe » ; PROCEED et la légende tiennent. Le rejeu de diagnostic de la
    note 15 n'a laissé que les lignes de promesse d'une reprise partielle (graine 9712), copiées dans
    `craft_v6_out/v84/qbert_check/` ; la comparaison mémoire n'a pas été conservée.
17. **Arrêts du superviseur pendant G1b (23-24 octobre).** 31 arrêts (mémoire engagée < 12 Go,
    sortie 3) de `v2_job_v84.py` ; chacun relancé de zéro par la file et terminé (sortie 0) ; aucune
    sortie partielle utilisée (JSON écrit en fin de run). Copie des lignes du superviseur :
    `craft_v6_out/v84/g1b_supervisor_actions.txt`. Débits G1b mesurés en temps mural ; le test 7 dirigé
    n'a jamais tourné (sans objet, aucun bras dirigé dans v84).
18. **Rapports G1b et G0-S : sens et pénalités (§3.2).** Sens de l'agent : pixels, ses boutons,
    max(0, r_step) par pas, ale.lives() ; fin de partie seulement comme limite d'épisode. L'échelle
    compte les hausses senties et ignore les pénalités ; sur les 22 jeux de place de développement,
    elle ne diffère du score vrai que sur carnival 9712 (6 820 contre 6 320), sans effet sur T1, C,
    λ ni k\*. Ces deux phrases manquent dans `room_verdict_v84.json` et `g0s/g0s_verdict.json` ; les
    comptes n'en dépendent pas. Le libellé G0-S RECIPE-INSENSITIVE vient de la note 13 (déclarée avant
    toute donnée), pas de la partie C (PASS-S / KC-search-sense) ; sa conséquence est celle de
    KC-search-sense. Ce n'est pas un négatif sur le mécanisme (instrument plafonné, note 14). Le κ de
    K\*A est au bord prolongé de la grille dans T8 et T8-S (leçon pour v85, pas un motif de relance).
19. **G0′ (rapport seulement, §6.1/§8) : non exécuté.** Il concerne la voie Σ, fermée par G0
    (RECIPE-INSENSITIVE) ; il ne peut ni rouvrir ni fermer quoi que ce soit. Déclaré non exécuté.
20. **Revue a posteriori du code G0-S (`wf_a259f054-320`, 24 octobre, étiquetée comme telle).** Aucun
    bogue n'a pu provoquer un faux échec de T8-S (chemin des données et chemin statistique relus). Lecture :
    dans le scénario positif, X:F = 1,146 ≈ K\*A:F = 1,156 — la structure commune plantée était captée
    presque au niveau de l'a priori parfait du jeu ; la limite est la vitesse de F, qui atteint la plupart
    des niveaux dès les premiers points de mesure (n₀ = 128 expansions ≈ 1,4 % de l'horloge), K\*A franchissant
    tout au premier point. Le plafond tient donc à la mesure (et au générateur planté, dominé par les axes de
    forte variance que F apprend en premier), pas au mécanisme. Écarts mineurs relevés : T8-S n'applique pas
    le plafond de 6 000 lignes de T8 (sans effet défavorable : il aurait plutôt abaissé le résultat) ;
    `planted_s.json` ne garde ni bornes ni détail par unité. Le libellé RECIPE-INSENSITIVE tient.
