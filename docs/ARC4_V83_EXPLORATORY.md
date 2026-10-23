# v83 — analyse exploratoire après KC-place (aucune revendication)

Écrit avant tout calcul (23 octobre). v83 s'est arrêtée à A1 (KC-place) ; A2 (G-préd) ne s'applique donc
pas. Cette analyse réutilise les journaux A1 (bras V, graines 9131-9132) pour une seule question de
conception : **ce qui précède une hausse de score (BON, 24 pas) ou une baisse de vies (MAUVAIS, 40 pas)
autour de « moi », appris sur d'autres jeux, se prédit-il sur un jeu tenu à l'écart mieux qu'un placebo
et qu'un taux de base par bouton ?** Le résultat sert uniquement d'entrée au panel de conception de la
version suivante ; ce n'est ni un test ni une preuve de transfert ni d'apprentissage plus rapide.

Réglages (déclarés ici, moins coûteux que ceux de G-préd) :
- lignes : au plus 5 000 lignes non perdues par run (graine 9149), 41 jeux ;
- jeux tenus à l'écart : les 8 jeux à place d'A1 (alien, amidar, bank_heist, ice_hockey, jamesbond,
  qbert, breakout, ms_pacman) ;
- W₋g, placebo W_perm,₋g (étiquettes décalées dans les lignes non censurées) et taux de base par bouton :
  exactement les fonctions de `scripts/train_promise_v83.py` et `scripts/gpred_v83.py` ;
- lecture rapportée telle quelle pour chaque jeu et tête : AUC, écart au meilleur des deux références,
  borne basse de l'écart (grappes de 50 itérations, 2000 rééchantillonnages) ; résumé = nombre de jeux
  où la tête passerait la règle G-préd (AUC ≥ 0,60 / 0,65, écart ≥ 0,03, borne basse > 0), à titre
  descriptif seulement.

Sorties : `craft_v6_out/v83/explore/` (magasin, W, `explore_promise.json`).

**Scellé (23 octobre, ~4 h).** Sur avis d'un juge du panel v84 : le premier test de v84 envisagé porte sur
les mêmes journaux A1 et une question voisine. Le résultat de cette analyse
(`craft_v6_out/v83/explore/explore_promise.json` et le journal de calcul au-delà des lignes de
convergence) n'est pas lu tant que les seuils de ce test v84 ne sont pas figés et committés. Le calcul
continue jusqu'au bout ; sa sortie sera lue et rapportée telle quelle après le gel.

**Amendement (23 octobre, gel du protocole v84).** Point de levée du scellé précisé : le résultat n'est lu
qu'après le commit de `g0_verdict.json` de v84 (`F:\dev\ragnarok_v84`), puis rapporté avec l'étiquette
« exploratoire ». À la fin du calcul, les SHA-256 de `explore_promise.json` et `run.log` sont committés
sans afficher leur contenu.
