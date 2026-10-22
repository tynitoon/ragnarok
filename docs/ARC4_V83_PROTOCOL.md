# ARC 4 v83 — « Carte des promesses » : apprend-il un jeu neuf plus vite grâce aux jeux d'avant ?
Protocole figé avant tout code d'agent (22 octobre 2026, ~6 h). Source : synthèse et juges
(`wf_9bcef103-94f`, copie `docs/panels/v83_promise_map_synthesis.md`, qui fait foi pour le détail
chiffré). Toute affirmation de v83 porte sur l'ENTRAÎNEMENT AVEC RETOURS EN ARRIÈRE (sauvegardes
facturées 60 images par chargement), sur Atari, actions non collantes, et avec les SENS du score et des vies.

## 0. Exception de Jérémie (22 octobre) et règles d'honnêteté
L'agent reçoit les HAUSSES du score (partie positive de la récompense ALE sur les 4 images d'un pas)
et le nombre de vies (`ale.lives()`), comme un humain qui lit ces chiffres. La fin de partie (`over`)
ne sert QUE de borne d'épisode : jamais d'étiquette, de variable ni de veto. Tout le reste (moi, effets
des boutons, objets, valeur) vient des pixels et des boutons. La RAM, le score total exact et l'état
interne restent côté évaluateur ; l'agent n'a que des poignées opaques de sauvegarde. Le code de
`ragnarok/agent_v83/` ne mentionne aucun nom de jeu (test AST/grep) et n'importe ni torch ni learner.

## 1. Ce qui est construit
- **Sens** (`SenseEnv`) : `step(a) -> (obs, hausse, vies, fin)` ; `clone()` -> poignée entière ;
  chaque restauration facturée 60 images ; registre facturé qui ne recule jamais (sondes et émulateur
  frère de l'adaptateur compris) ; trace évaluateur (images facturées, score cumulé vrai) illisible
  par l'agent.
- **Entrée** (identique pour tous les bras) : fond appris sur 12 000 pas, adaptateur de boutons par
  fabrique, amorçage de « moi » v74, contrôle de perte de moi. Pas de recherche de compteurs.
- **Explorateur V** (portage de l'explorateur v71) : case = (moi sur grille 16×16 ou (−1,−1) ;
  vies ; floor(4·log2(1+S))) avec S = hausses cumulées le long de la trajectoire ; meilleure case =
  S plus grand puis profondeur plus courte ; choix pondéré v71 vectorisé ; déroulés de 100 pas, bouton
  uniforme tenu (répétition 0,95) ; une baisse de vies termine le déroulé sans archivage ; sondes au
  plus une tous les 8 pas quand je suis perdu ; plafond 20 000 cases (éviction déclarée : la plus
  choisie sous le meilleur S, puis la plus ancienne) ; arrêt dur au plafond d'images facturées.
- **Promesses** (bras F, X, P, X0, K*) : vue 64×64 autour de moi (avant-plan sans couleur, moi
  masqué ; objets, apparus, disparus, petits objets), tournée pour que le mouvement du bouton candidat
  pointe vers le haut ; ~430 variables ; deux têtes logistiques : BON = une hausse sous 24 pas,
  MAUVAIS = une baisse de vies sous 40 pas (étiquettes venant des sens seulement) ; ordonnées libres
  par jeu ; réajustement tous les 250 itérations sur un réservoir ≤ 120 k lignes ; perte = log-perte +
  (n0/(n0+n))·(κ0/2)·‖w − w0‖², n0 = 50 000, κ0 = 1. Pilotage : (a) choix de case parmi 16 candidats
  tirés comme V, proportionnel à exp(β·clip(z, −2, 2)) ; (b) bouton tiré selon (1−α)·uniforme +
  α·softmax(logit BON − λ·logit MAUVAIS) ; β = 1, α = 0,5, λ = 1 ; avec w = 0, exactement V (testé).
- **Objet transporté Π** = poids des têtes (hors ordonnées) + échelles. W_dev : tous les jeux de
  développement et de calibration, poids égal par jeu ; W₋g : sans le jeu g ; W_same(g) : le jeu g
  seul ; placebo W_perm : étiquettes décalées en cercle d'au moins 10 déroulés dans chaque jeu (graine
  9149) puis normes égalisées à W (garde taux de base, regroupement temporel et confiance ; casse le
  lien état → issue). Si une seule tête passe G-préd, seule elle est transportée.
Bras : V (sans promesses) ; F (w0 = 0, neuf) ; X (W₋g en développement, W_dev au test scellé) ;
P (placebo) ; X0 (W figé, attribution) ; K* (W_same, développement seulement : plafond de valeur).

## 2. Lots et graines
Lots (repris du juge v82, `ragnarok/evaluation/pools_v83.py`) : développement 35 jeux ; calibration
6 jeux ; test scellé 24 jeux ; second tour 9 jeux ; jamais ouverts : boxing, frostbite, riverraid,
kangaroo ; enduro exclu. Graines (vérifiées au §9) : 9130 rodage (jamais une mesure) ; 9131-9132
place (V) et journaux sources ; 9133 barres de développement ; 9134 diagnostic ; 9135-9138 VALEUR et
gel 1 (appariées entre bras) ; 9141-9144 gel 2 ; 9149 décalage placebo et tirage des unités ; réserve
9139, 9140, 9145, 9147, 9148 ; scellé : 9230 barres, 9231-9232 éligibilité, 9233-9236 bras ;
9241-9242 diagnostic en direct ; second tour : 9250 / 9251-9252 / 9253-9256.

## 3. Barres, paliers, montées (côté évaluateur)
Barre B_g = max(meilleur bouton constant, hasard uniforme, hasard répété 0,95), 10 épisodes chacun,
non collant, 0-30 NOOP, 7000 pas. Pas u_g = max(0,5·|B_g|, médiane des hausses positives vraies des
épisodes de référence). Paliers M_k = B_g + 2^k·u_g, k = 0..10. T_s(M) = images facturées depuis
l'image 0 (entrée comprise) au premier clone dont le score vrai cumulé ≥ M, confirmé par rejeu depuis
la racine (écart = Interruption). Montée C = T(M*) − T(M1) depuis le M1 propre à chaque bras ; montée
tardive C_tard = T(M*) − T(M2). Censure toujours CONTRE le bras transporté (X, X0, K*) : non-atteinte
de M* = plafond ; comparateur (V, F, P) = plafond − T(M1) ; plancher C ≥ 5 000. ℓ_g = moyenne log C
(comparateur) − moyenne log C(X) ; r_g = exp(ℓ_g) borné [1/8 ; 8] ; R = exp(moyenne des ℓ_g) ;
IC 95 % par rééchantillonnage à deux niveaux (jeux, puis graines par bras ; 2000).

## 4. Étape A (développement)
**A1 — G-place** (41 jeux, bras V, graines 9131-9132, plafond 2 M) : jeu « à place » si (a) chaque
run a ≥ 10 hausses dans ses 300 k premières images, (b) M*_g (plus haut palier atteint par les deux
graines sous 2 M) ≥ M3, (c) médiane T(M1) ≤ 1 M, (d) médiane C ≥ 300 k. Pas de filtre de stabilité
par jeu ; dispersion poolée σ̂² = moyenne des (log C₁ − log C₂)²/2. G-place réussi si ≥ 13 des 35 jeux
de développement ont de la place ET se_proj = sqrt((σ̂²/2 + 0,09)/G_proj) ≤ 0,23 avec
G_proj = 17·n_place/35 (24 jeux scellés moins 7 jeux d'exploration difficile supposés inéligibles :
montezuma_revenge, pitfall, venture, private_eye, solaris, gravitar, skiing). Sinon KC-place :
« avec retours en arrière et les sens, l'entraînement neuf de cet explorateur est trop bon marché,
saturé ou bruité pour qu'une montée 2× plus rapide soit mesurable sur le lot scellé » ; la carte des
coûts par jeu est gardée. `room_verdict.json` committé avant toute image F, X, P, K*.
**A2 — G-préd** (hors ligne) : sur chaque jeu à place, W₋g noté sur les lignes A1 du jeu (grappes de
50 itérations). Une tête passe sur g si AUC ≥ 0,60 (BON) ou ≥ 0,65 (MAUVAIS) ET ≥ +0,03 au-dessus du
meilleur de {taux de base par bouton ajusté sur g, placebo W_perm,₋g}, borne basse > 0 ; elle passe si
c'est le cas sur ≥ 50 % des jeux à place et au moins 6. Sinon KC-préd. Rapporté seulement (prédiction
écrite) : l'avantage de l'a priori tombe sous 25 % de sa valeur initiale vers 50 k lignes.
**A3 — VALEUR** (jeux à place, au plus 12 tirés par la graine 9149 ; V, F, K* ; 4 graines 9135-9138 ;
plafond 4 M) : R(K* contre le meilleur de V et F) ≥ 2,0, borne basse ≥ 1,25, r_g ≥ 2 sur ≥ 1/3 des
unités. Sinon KC-valeur. Drapeau KC-nuisance rapporté (F plus de 1,25× plus lent que V).
**A4 — répétition** (au plus 2 gels) : gel 1 = X (W₋g) et P (W_perm,₋g), graines 9135-9138,
appariés à V et F d'A3 ; X0 9135-9136. GO si R(X:F) ≥ 1,5 borne basse > 1,1 ; R(X:V) borne basse > 1 ;
R(X:P) borne basse > 1 ; R_tard(X:F) borne basse > 1. Sinon un gel 2 déclaré avant ses runs, ne
touchant QUE la machinerie des promesses (variables, horizons, β, α, λ, n0, κ0, calendrier de
réajustement ou petite tête MLP), jamais les sens, l'explorateur, la case, les paliers ni les seuils ;
re-run 9141-9144 ; nouvel échec = KC-dév (le lot scellé reste fermé). Toute modification du cœur de
l'explorateur après A1 invalide A1 (Interruption, A1 refait sur graines de réserve).

## 5. Étape B (test scellé, une fois)
B0 : barres (9230) puis runs d'éligibilité V (9231-9232, plafond 2 M) sur les 24 jeux ; règle de
place A1 ; `sealed_eligibility.json` committé avec son empreinte avant tout bras. Moins de 6 unités :
second tour une fois ; toujours moins de 6 : NON MESURABLE.
B1 : X (W_dev), F, V, P (W_perm), 4 graines appariées 9233-9236 ; X0 9233-9234 ; plafond 4 M.
Mesures : R(X:F), R(X:V), R(X:P), R_tard(X:F) ; avance au départ HS = moyenne géométrique de
T_F(M1)/T_X(M1) ; p unilatéral par permutation des étiquettes X/F dans chaque jeu ; minimum de R(X:F)
en laissant un jeu de côté ; ligne d'instrument : F 9233-9234 contre 9235-9236, rapport dans [0,7 ; 1,43].
Verdicts (le premier qui s'applique) : 1 NON MESURABLE ; 2 NON CONCLUANT (instrument hors bornes, ou
≥ 50 % des runs X ou F censurés) ; 3 PLUS VITE-2× : R(X:F) ≥ 2,0 borne basse ≥ 1,3 et p ≤ 0,025 ;
R(X:V) ≥ 1,5 et R(X:P) ≥ 1,5 bornes basses > 1 ; R_tard(X:F) ≥ 1,5 borne basse > 1 ; r_g(X:F) > 1 sur
≥ 2/3 des unités ; minimum en laissant un jeu de côté ≥ 1,5 ; 4 PLUS VITE-PARTIEL : bornes basses de
R(X:F), R(X:V), R(X:P), R_tard(X:F) toutes > 1 et p ≤ 0,025 ; 5 TRANSFERT NÉGATIF : borne haute de
R(X:F) ou R(X:V) < 1 ; 6 AVANCE SEULEMENT : HS ≥ 1,5 borne basse > 1 sans 3 ni 4 ; 7 NUL.
Légendes : PILOTAGE GÉNÉRIQUE (bornes basses de R(X:F) et R(P:F) > 1 mais celle de R(X:P) ≤ 1) ;
attribution (R(X0:F) ≥ 0,8·R(X:F) : « a priori fixe des jeux d'avant », sinon « a priori + apprentissage
dans le jeu »). Écrit d'avance : un vrai effet 2× ne passe le seuil sur l'estimation qu'environ une fois
sur deux. Rapporté aussi : images jusqu'à chaque palier, aire sous log score / log images, censures,
part des sondes, diagnostic en direct D2 (sans sauvegarde, collant 0,25, graines 9241-9242).

## 6. Arrêts, budget
KC-place, KC-préd, KC-valeur, KC-dév ; placebo intégré à tout verdict PLUS VITE ; défaut en cours de
file = Interruption, résultats partiels committés ; défaut après notation jamais renoté sur ce lot.
Budget : 14 jours calendaires ou 2,0 G d'images facturées (développement plafonné à 1,3 G) ; le test
scellé commencé est terminé. 2 travailleurs (3 quand Tavern est calme), ≤ 3 Go chacun, sous le
superviseur, gardes à 14 Go de RAM et de mémoire réservée, sorties avec provenance.

## 7. Ce que v83 n'établit pas
Aucune compétence en direct (le mur de la distillation reste) ; aucune perception du score ni de la
mort (données par les sens) ; pas « n'importe quel jeu » (seulement ceux à place) ; aucune avance au
départ présentée comme apprentissage plus rapide ; rien sur NES, 16 bits ou PC ; un PLUS VITE dirait
seulement qu'un a priori état → issue transporté accélère la montée de CET explorateur.

## 8. Chances, dites honnêtement
Rodage ≈ 0,85 ; G-place ≈ 0,45 ; G-préd sachant place ≈ 0,45 ; VALEUR ≈ 0,35 ; GO ≈ 0,4 ; ≥ 6 unités
scellées ≈ 0,7 ; PLUS VITE-2× sachant B mesurable ≈ 0,35. Ensemble : PLUS VITE-2× ≈ 0,01 ; PARTIEL ou
mieux ≈ 0,02 ; réponse nette à G-place ≈ 0,85. Issue la plus probable : un négatif propre en 4 à 6 jours.

## 9. Vérification des graines (à remplir à A0, avant toute image)
Recherche faite le 22 octobre (~6 h) dans `ragnarok/`, `scripts/`, `docs/`, `tests/` de toutes les
copies (`ragnarok`, `_v73`, `_v74`, `_v75`, `_v80`, `_v81`), motifs « seed… 91xx/92xx » et « _s91xx/_s92xx »
pour 9130-9149 et 9230-9256 : aucune occurrence en contexte de graine. Recherche numérique large : seule
occurrence 9248 comme seuil de points dans `ragnarok/learning/entry_trial_plan_v70.py` (pas une graine,
et 9248 n'est pas utilisée par v83). Les graines v82 (9101-9111, 9201-9211) restent intactes.
