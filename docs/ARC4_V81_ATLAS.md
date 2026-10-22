# ARC 4 v81 — « L'atlas d'entraînement » : la place pour apprendre avant le mécanisme
Protocole écrit et figé avant tout code d'agent (22 octobre 2026, 3 h). Source : post-mortem
contradictoire de v80 et verdict du juge (`wf_2d3f677e-77a`). Exploratoire ; chaque « plus vite » obtenu
avec retours en arrière est étiqueté comme tel et séparé du jeu en direct.

## 1. Pourquoi
v72-v80 ont buté sur deux murs : distiller des valeurs de long horizon en réflexe (v72-v75), et une
couche d'objets fragile sur laquelle bâtir une théorie (v80). Surtout, aucune étape n'a mesuré si
« apprendre plus vite » est seulement VISIBLE sur ces jeux : si un agent neuf atteint ses paliers de
score presque gratuitement grâce aux retours en arrière, aucun gain de vitesse ne pourra se voir
(c'est ce qui a tué l'arc 2, KC5). v81 mesure d'abord cette place, avec une carte simple valable pour
tout jeu — où je suis × ce que mes actions font là — bâtie sur la perception qui tient depuis v71.

## 2. Ce qui est construit
- **Atlas** : cases (ma position grossière × état des compteurs), macros d'effet terminées par un
  événement (au plus 16 pas), arêtes mesurées par branches sœurs (où j'arrive, mort, points).
- **Point de non-retour** : le début de ma mort daté par la divergence entre branches sœurs qui meurent
  et qui survivent (mes propres étiquettes, jamais la RAM) ; sert d'horloge et de profondeur de rembobinage.
- **Moi = ce que mon bouton change** (segmentation contrefactuelle, toutes mes couleurs) : seule correction
  générique permise de la perception de moi.
- **A priori d'affordance** : modèle simple (logistique, numpy) sur la vue autour de moi + l'effet
  demandé → bloqué / mort sous 16 pas ; admis seulement s'il bat sur la perte hors échantillon une table
  sans pixels (effet × position).
- **Planification** : itération de valeur sur l'atlas, exécution en boucle fermée, veto.
Rien dans `ragnarok/agent_v81/` ne dépend du nom du jeu (vérifié par grep à la revue).

## 3. Graines, départs, jeux
Graines (vérifiées inutilisées) : 8101/8102/8103 entraînement et pratique ; 8104 validation (itération
libre) ; 8105 test scellé, noté une fois par code figé ; 8106 réserve. Les graines 20/21, 40/41 et les
jeux réservés restent intouchés. Départs : au moins 40 grappes indépendantes par partie, chacune
commençant par 0-30 NOOP aléatoires puis 0-2000 pas d'échauffement aléatoire répété, flux dépendant de
la graine ; dédoublonnage par la RAM côté évaluateur seulement ; au plus 6 racines par grappe, espacées
d'au moins 16 pas ; intervalles par rééchantillonnage des grappes (2000).
Candidats (fixés maintenant) : Bank Heist, Wizard of Wor, Amidar, Crazy Climber, Freeway.

## 4. Seuils (fixés maintenant)
**E — éligibilité (évaluateur, graines d'entraînement)** : lecteur de score précision et rappel ≥ 0,9
contre la RAM ; signal de mort de l'agent précision ≥ 0,9 ; l'apprenant v71 existant, avec 2 M d'images
facturées (graine 8101), atteint en pratique un meilleur score ≥ 2 × max(meilleur bouton constant,
hasard répété). Prédiction écrite : Freeway échoue (25 contre 21 < 2×). Moins de 3 jeux éligibles : KC-pool.
**G0-moi** (chaque jeu éligible) : moi à ≤ 4 px du juge RAM sur ≥ 90 % des pas (base et hasard, pas de
replacement exclus) ; mort précision ≥ 0,9 et rappel ≥ 0,7 ; score précision ≥ 0,9 ; début de mort estimé
à ±3 pas du point de non-retour de l'évaluateur (tiré de lives(), ≤ 100 morts par jeu) sur ≥ 70 % d'au
moins 30 morts. Sur ≥ 3 jeux éligibles ; au plus 2 corrections génériques, identiques pour tous les jeux.
**G-pré** (partie tenue à part des grappes d'entraînement ; bloque la suite) : calibration par décile
des issues des arêtes, pente dans [0,7 ; 1,3] pour P(mort) et E(points) ; aucune arête d'au moins 50
visites dont le taux de mort prédit s'écarte de plus de 0,2.
**G-place** (atlas neuf, graines 8101/8102/8103, plafond 2 M d'images facturées chacune) : paliers
M1..M4 = 1,5 / 2 / 4 / 8 × max(meilleur constant, hasard répété) sur les départs d'entraînement ; M* =
plus haut palier atteint par les 3 graines sous le plafond ; un jeu passe si M* ≥ M2, médiane des images
jusqu'à M* ≥ 200 k, et rapport max/min entre graines ≤ 2,0. Sur ≥ 2 jeux.
**G-a priori** (hors ligne, sur les journaux de pratique, sans nouvelle image) : modèle d'affordance noté
sur les grappes de validation, seulement sur des couples (case, effet) dont la case n'apparaît jamais
dans les journaux d'ajustement : AUC(bloqué) ≥ 0,75 et AUC(mort sous 16) ≥ 0,65 ; chacun ≥ +0,05
au-dessus du meilleur de {taux de base par effet, table effet × case de position, placebo à étiquettes
permutées}, borne basse par grappes > 0 ; sur ≥ 2 jeux. Laisser-un-jeu-de-côté : rapporté seulement.
**G-direct** (validation 8104 pendant le développement, puis 8105 scellé une fois) : sans sauvegarde,
actions collantes 0,25, 10 parties par graine d'évaluation, 7000 pas. REF = max(meilleur constant,
hasard répété, apprenant v71 avec les mêmes images facturées). Chaque graine de pratique bat REF de
plus de 2·sqrt(se_graine² + se_ref²) sur ≥ 2 jeux. Un échec ne bloque que les affirmations « en direct ».
**Entrée en A2** (transfert proche vers des régions ou niveaux plus tardifs, départs donnés par
l'évaluateur) : E, G0-moi, G-pré, G-place et G-a priori dans le jeu réussis ; protocole A2 figé avant
toute image A2 ; bras X, X0 (a priori figé), P (placebo), quatre neufs Fa-Fd (graines ≥ 8110) ;
affirmation : coût de montée de X de M1 à M* ≤ 0,5 × celui de chaque neuf et ≤ 0,5 × celui de P, IC par
grappes, sur ≥ 2 jeux.

## 5. Critères d'arrêt (fixés maintenant ; négatifs gardés ; pas de relance sur les mêmes graines)
KC-pool (< 3 jeux éligibles : arrêt sur ce lot) ; KC0 (G0-moi échoue sur ≥ 2 jeux éligibles après 2
corrections génériques) ; KC-pré (G-pré échoue après 1 correction générique : les statistiques d'arêtes
ne sont pas un modèle) ; KC-place (pratique bon marché, médiane < 200 k, ou bruitée, rapport > 2,0, sur
tous les jeux sauf au plus un : « avec retours en arrière, la pratique Atari est trop bon marché ou trop
bruitée pour montrer 2× plus vite » ; pas d'A2) ; KC-cases (sur ≥ 2/3 des jeux, rien au-delà de M1 sous le
plafond) ; KC-a priori (G-a priori échoue sur les cases inédites : pas d'A2) ; KC-direct (affirmations
« en direct » seulement) ; KC-avance (A2 : X seulement en avance au départ = avance, pas apprentissage
plus rapide).

## 6. Règles de budget et anti-terrier
A1 plafonné à 7 jours calendaires et 40 M d'images facturées. Outils d'évaluation : au plus 1,5 jour ; un
jeu dont le juge RAM de moi n'est pas validé à temps est abandonné, pas prolongé. Au plus 3 gels déclarés
sur la validation ; une notation scellée par seuil, plus une sur la réserve 8106 avec défauts nommés
d'avance. Toute sortie porte sa provenance (empreinte du code, des entrées, heure) ; une sortie périmée
est refusée. Un processus < 3 Go, sous le superviseur RAM.

## 7. Chances, dites honnêtement
E sur ≥ 3 jeux ≈ 0,6 ; G0-moi ≈ 0,55 ; G-pré ≈ 0,75 ; G-place ≈ 0,45 ; G-a priori sur cases inédites ≈ 0,5 ;
tout A1 ≈ 0,2 ; A2 sachant A1 ≈ 0,3 ; « apprend plus vite » entre jeux Atari ≤ 0,1. Issue la plus
probable : KC-place ou un A1 partiel — une réponse propre et bon marché à la question qui compte.

## 8. Choix fait au nom de Jérémie (il peut le changer)
Sur Atari, « plus vite » est d'abord mesuré comme coût d'entraînement AVEC retours en arrière, étiqueté
comme tel ; Assault et Freeway sont abandonnés comme terrains de test (KC1 ; « monter » quasi optimal).
EOF
cd F:/dev/ragnarok_v81 && git add docs/ARC4_V81_ATLAS.md && git commit -q -m "ARC 4 v81 'practice atlas' protocol frozen before any agent code: measure first whether learning faster is visible at all (G-room), atlas of where I am x what my actions do, affordance prior on unseen cells, new seeds 8101-8106, gates, kill criteria, budget

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1