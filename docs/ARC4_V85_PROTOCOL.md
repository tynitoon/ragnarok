# ARC 4 v85 — « Le savoir a-t-il de la place dans la montée ? » (protocole, version française)

Figé le 3 novembre 2026, avant tout code et tout calcul de v85. **Le texte qui fait foi est l'anglais :
`docs/ARC4_V85_PROTOCOL_EN.md`** (plan final du panel `wf_8db018d6-25c` : 5 propositions, 3 juges,
synthèse, 3 audits, version finale ; copies dans `docs/panels/v85_*`). Ce document le résume et s'y soumet.

## En-tête de toute affirmation
Toute affirmation de v85 porte sur une RECHERCHE AVEC RETOURS EN ARRIÈRE FACTURÉS (sauvegardes, 60 images
par chargement, chaque image émulée facturée), avec les SENS accordés le 22 octobre 2026 (hausses du score,
nombre de vies). Ce n'est pas du jeu en direct. Les instruments « oracle » (O_4, K\*map, K\*mac, D) sont des
outils de l'évaluateur, jamais des agents.

## Ce que fait v85, dans l'ordre
1. **S0** (aucune image facturée) : reconstruit, pour chaque jeu à place de G1b, des départs partagés s1
   (état au palier M1 sur une lignée de V2 qui a atteint le palier du haut), met en cache l'entrée, fixe
   les paliers comptés et un plancher physique.
2. **Étape I** (V2s et D seulement) : mesure le bruit du régime à départ partagé et y plante des
   accélérations exactes (×1, ×1,25, ×1,5, ×2, tardive seule, saut au premier palier) ; la règle de
   calibration écrite d'avance en tire n, τ_pt et τ_lb (F1). « Pas de place » exige une borne haute sous 2.
3. **G1** (en ligne) : trois instruments — O_4 « quelle proposition payer » (un coup d'œil gratuit par
   proposition, puis la gagnante est jouée sur un hasard neuf), K\*map « où repartir », K\*mac « comment
   bouger » — mesurent si un savoir du même jeu fait grimper V2 environ 2× plus vite depuis un palier
   commun. Une voie n'ouvre une famille transportable que si la famille peut porter cette information
   (W : a priori de critique d'expansion ; M : vocabulaire moteur).
4. **G2a / G2b** (développement, une fois chacun), puis **un seul test scellé** sur des jeux jamais ouverts.

Tout le code des familles, chaque objet transporté, témoin et placebo est écrit, testé et haché avant G1.

## Graines (vérifiées le 3 novembre)
Bloc 9800-9899 (table §12 du texte anglais). `git ls-files` des dépôts `ragnarok`, `ragnarok_v83`,
`ragnarok_v84` (environnements virtuels exclus), motif « seed … 98xx » : aucune occurrence. Les numéros
9712-9713, 9749 et 9256 cités par le plan désignent des données ou lots antérieurs, pas de nouvelles graines.

## Chances (texte anglais §14)
De la place quelque part ≈ 35 % (≈ 50 % si G1 tourne) ; gain transporté confirmé ≈ 1 % ; 2× confirmé ≈ 0,3 %.
Si la réponse est non : résultat net et borné ; suite par défaut, les jeux où la recherche se bloque.

## Décision réservée à Jérémie
Seul cas : une mesure jugée sous-puissante à l'étape I (« underpowered ») ; sinon rien à décider.
