# ARC 4 v80 — l'agent qui se fait une théorie du jeu (conception, 21 octobre 2026)

Statut : conception figée avant tout code d'agent. Exploratoire. Les seuils ci-dessous sont fixés
maintenant et ne changeront pas après observation. Source : panel de conception du 21 octobre
(inventaire, 4 concepteurs, 3 juges, synthèse ; journal `wf_62afcea2-4cb`) et consigne de Jérémie :
« innove, sois rigoureux, avance par méthode scientifique et instinct ».

## 1. Pourquoi changer de principe
v72-v75 ont établi que l'agent perçoit bien (moi, boutons, compteurs, mort lue à l'écran) et que
ses expériences en laboratoire mesurent de vraies valeurs ; mais un réflexe appris (réseau sur
pixels) n'en retient presque rien : au niveau du meilleur bouton constant en direct, ~10 % de
l'oracle hors ligne sur Assault, rien sur Demon Attack, transfert nul. Un réflexe demande des
millions d'exemples et son savoir, enfermé dans des poids, ne se transporte pas.

## 2. Ce qui est neuf (et ce qui ne l'est pas)
Les « world models » existants (Dreamer, MuZero, IRIS…) prédisent des pixels ou un état latent
appris par récompense sur des millions d'images, traitent le jeu comme une boîte noire jouée en
avant, et gardent leur savoir dans un réseau opaque. v80 fait trois paris différents :
1. **Le laboratoire à remonter le temps.** Depuis une sauvegarde, l'agent essaie chaque bouton
   (branches sœurs) : la différence entre branches, proche de moi ou de mes tirs et peu après mon
   action (« cône causal » : 8 pas, 16 px), est une INTERVENTION, pas une simple observation. Un
   humain qui joue n'a jamais cela ; l'apprentissage par renforcement classique ne s'en sert pas.
2. **Une théorie lisible et notée.** Le savoir est un livre de règles par type d'objet (petits
   arbres JSON : mouvement, apparition, disparition, issue d'un contact avec moi ou avec mes tirs,
   jauges). Avant chaque expérience, la théorie prédit ; le laboratoire note ; on ne garde que ce
   qui prédit mieux sur des données jamais vues (sélection préquentielle). Le livre se lit.
3. **Le savoir transporté est un savoir de rôles, vérifiable à bas prix.** Ni couleur, ni bouton,
   ni identifiant de jeu : des signatures de comportement → issues probables (« ce qui descend vers
   moi en ligne droite tue »), puis une bibliothèque de fragments de règles. Hypothèse de fond :
   **vérifier une règle connue coûte quelques expériences, la découvrir en coûte beaucoup** ; c'est
   exactement la condition sous laquelle les arcs précédents ont vu la réutilisation payer.
En jeu réel (sans sauvegarde), l'agent **imagine** : toutes les 8 étapes, il simule chaque bouton
avec sa théorie (particules) et ne quitte son bouton de base que si l'imagination est nettement
d'accord ; s'il est trop souvent surpris par le jeu, il le sait et revient à sa base.

Ce qui n'est pas neuf : la perception (échafaudage classique de v71, identique dans tous les bras,
déclaré, jamais partie d'une revendication de transfert), l'appariement de suivi (hongrois), les
modèles de mouvement simples. La nouveauté est la boucle intervention → règle notée → imagination,
et ce qui se transporte.

## 3. Architecture (milestone 1)
- **Échafaudage (v71, inchangé)** : fond, compteurs et événements, moi (sondes + suivi), boutons,
  signal de mort (compteur de vies OU « disparu 2 pas »). `over` (fin de partie ALE) n'est jamais
  une étiquette.
- **Objets** (`agent_v80/objects.py`) : composantes 8-connexes par couleur, fond retiré, compteurs
  et jauges masqués, au plus 48 ; suivi hongrois (prédiction par la règle courante, porte 16 px,
  4 pas sans observation) ; marques MOI, MIEN (mes tirs), DÉCOR.
- **Théorie** (`agent_v80/theory.py`) : types (classe de couleurs × taille × forme, ≥ 30 vues) ;
  par type : règle de mouvement choisie dans {immobile, vitesse constante, rebouclage, rebond,
  vers moi, accélération, marche} par vraisemblance préquentielle à 8 pas (bruit de Laplace + 2 %
  d'aberrants) ; règle d'apparition (parent : moi par bouton, autre type, bord) ; registre des
  contacts contre MOI et MIEN (Dirichlet sur {mort, point, disparaît, je suis déplacé, rebond,
  rien}, interventions et observations comptées à part) ; signature de comportement à 16 nombres ;
  ma carte de déplacement par bouton ; jauges (remplissage, dérive par bouton, risque) ; feuille de
  valeur (ma case 16×16 × quartile de jauge → mes événements sur 32 pas) ; journal préquentiel.
- **Laboratoire** (`agent_v80/lab.py`) : racines uniformes sur les trajectoires de la base, chaque
  bouton tenu 8 pas puis la base 56 pas (sans actions collantes) ; la théorie prédit d'abord, le
  jeu note ; attribution par branches sœurs dans le cône causal ; carte de différences pour
  découvrir les jauges ; une mort inexpliquée répartit le blâme entre les suspects proches.
- **Imagination** (`agent_v80/imagine.py`) : 16 particules par bouton, tenir 8 puis la base 24,
  valeur = E[points] − 2·P(mort) + feuille ; écart si gain ≥ 0,25 et ≥ 80 % des particules d'accord ;
  décision tenue 8 pas ; auto-contrôle de surprise. Garde de confiance côté agent : si sur ses
  propres racines de calibration le choix imaginé ne bat pas la base, il joue sa base.
- **Évaluation** (`agent_v80/evaluate.py`) : seul lecteur du vrai score ; RAM côté évaluateur
  seulement (juges calibrés `ragnarok/evaluation/*_v70.py`).
- Numpy seul, déterministe, sans GPU. Un processus, < 3 Go, gardes RAM et commit à 14 Go.

## 4. Étapes et seuils (fixés maintenant ; une seule variante pré-déclarée : H = 16)
Graines de développement 98/99 seulement ; protocole figé : entraînement 20/21, évaluation 40/41
(vérifiées inutilisées avant le gel). Racines d'évaluation hors ligne tenues par l'évaluateur
(300 depuis des départs neufs, sans jumeau d'écran des racines d'entraînement).
- **G0 perception** : Freeway, ≥ 90 % des voitures appariées à ≤ 4 px de la RAM ; Assault, ≤ 30 %
  de mes pas perdus ou fusionnés. Contrôle des étiquettes : précision ≥ 0,9 des morts et points de
  l'agent contre la RAM (sinon STOP : la théorie apprendrait du faux).
- **G1 compréhension** : Assault, classement des morts entre branches sœurs ≥ 0,70 (borne basse
  > 0,5, ≥ 30 paires), score de Brier de P(mort sous 32) ≥ 0,10 au-dessus de la climatologie par
  bouton, idem pour les points ; Freeway, position imaginée des voitures à 16 pas à ≤ 4 px pour
  ≥ 80 % et erreur ≤ 0,8 × l'extrapolation à vitesse constante.
- **G2 décision hors ligne** : sur Assault, la valeur réelle (64 pas) du choix de la théorie dépasse
  la meilleure déviation constante de ≥ +0,10 (IC > 0) ET le réflexe `conv_q` entraîné sur les
  mêmes racines de ≥ +0,05 (IC apparié > 0).
- **G3 direct, développement** (graines 98/99) : sur Assault OU Freeway, battre REF de plus de
  2 erreurs types combinées ; REF = max(chaque bouton constant, hasard, hasard répété, agent non
  entraîné, meilleur agent Ragnarok précédent) sur les mêmes parties. Prédictions écrites :
  Freeway ≥ 23 (constant ≈ 21) ; Assault ≥ constant + 40.
- **G4 protocole figé** (seulement si G3) : 6 jeux, chaque graine d'entraînement bat REF de plus de
  2 erreurs types combinées, au moins 2 jeux sur 6.
- **Transfert (plus tard, conditions : G4, ou G1 sur ≥ 3 jeux sources pour un résultat « niveau
  connaissance »)** : un jeu laissé de côté à tour de rôle ; bras X (rôles + bibliothèque),
  X0 (a priori seul, figé), Fa/Fb (neufs), P (placebo : rôles permutés), F10 (neuf avec 10× de
  calcul) ; courbe de compréhension en fonction des images de laboratoire facturées ; coût de
  montée entre niveaux appariés (le niveau de départ est celui de X0) ; « apprend plus vite »
  seulement si X monte au moins deux fois moins cher que chaque neuf ET que le placebo, sans finir
  plus bas, et par type d'objet. Le correcteur doit rejeter une simple avance simulée.

## 5. Critères d'arrêt (fixés maintenant ; négatifs gardés, pas de relance sur les mêmes graines)
KC0 perception (G0 échoue après 2 corrections déclarées) ; KC-étiquettes (< 0,9) ; KC1 compréhension
(Assault : classement < 0,60 et Brier < 0,05 après la variante H = 16) ; KC2 valeur (G2 échoue :
« une théorie d'objets ne décide pas mieux que le réflexe à coût égal ») ; KC3 écart labo/direct
(G2 réussi, G3 échoue sur les deux jeux, une correction permise) ; KC4 calcul (> 50 ms par
décision après une optimisation) ; KC5 bon marché (les agents neufs atteignent 90 % de leur niveau
en 40k images sur ≥ 3 jeux sur 4 : pas de revendication de transfert sur Atari) ; KC6 placebo ;
KC7 simple avance.

## 6. Chances, dites honnêtement
G0 ≈ 0,75 ; G1 (Assault) ≈ 0,5 ; G2 ≈ 0,35 ; G3 sur un jeu ≈ 0,3 ; G4 ≈ 0,15 ; « apprend plus
vite » sur Atari ≤ 0,1. Issue informative la plus probable : la théorie comprend et bat le réflexe
hors ligne sur Assault, petit gain en direct sur Freeway ou Assault, nul sur Demon Attack, et KC5
ou KC7 pour le transfert — ce qui laisse un instrument lisible à emporter vers la NES.

## 7. Après le milestone 1 (pour mémoire, non engagé)
Milestone 2 : le scientifique choisit ses expériences (racines où sa théorie est la plus incertaine,
tests de contact ciblés) au lieu de racines uniformes — mesuré contre le tirage uniforme à images
égales. Milestone 3 : bibliothèque de fragments de règles (longueur de description minimale) et
expérience de transfert.

## 8. Revue du code avant le premier lancement (21 octobre) et écarts du milestone 1
Revue contradictoire (`wf_a6de8661-f12`, 4 angles, sceptiques) : 3 défauts invalidants corrigés
avant tout calcul (l'image de départ passait deux fois dans le suivi : vitesses nulles ; image
précédente fausse pour le réflexe de comparaison ; vérité évaluateur arrêtée à la mort), et des
défauts majeurs corrigés (naissances sans vitesse ; issue d'un contact retirée à chaque pas ;
futurs non appariés entre boutons ; feuille de valeur sans les morts ; intervalles non regroupés
par trajectoire ; référence climatologique prise sur l'entraînement ; branches censurées comptées
comme survivantes ; tirs « à moi » définis par type au lieu de par objet ; délai de recharge
calculé à travers les racines ; règle de mouvement choisie sur les mêmes données).
Fenêtres du contrôle des étiquettes déclarées maintenant : mort juste si la RAM montre une vie
perdue ou une fin de partie dans [t−8, t+60] ; point juste si le vrai score monte dans [t−4, t+4] ;
au moins 20 étiquettes, sur le test ET sur un échantillon de 60 racines d'entraînement.
« D'accord » dans la règle de décision : particule appariée au moins aussi bonne que la base (≥).
Écarts au §3 assumés pour le milestone 1 (à noter dans tout résultat) : issues de contact
{mort, point, disparaît, je suis déplacé, rien} (pas de « rebond ») ; mouvements {immobile, vitesse
constante, rebouclage, rebond aux bords, vers moi} (pas d'accélération ni de marche) ; pas de
parents de naissance, pas de répartition du blâme pour les morts inexpliquées, pas de signature
à 16 nombres (milestone 3) ; choix de règle par erreur médiane à 8 pas sur la seconde moitié des
traces (au lieu de la vraisemblance de Laplace). G0 Assault n'a pas de juge RAM : il est jugé
contre le suivi de l'agent lui-même (dit dans le résultat). Le réflexe de comparaison (conv_q)
garde l'entrée v74, dont la vue globale ne voit pas la jauge d'Assault (dit dans le résultat).
Graines : les caches de développement sont refaits en graine 98 ; le collecteur refuse toute
autre graine hors protocole figé.

## 9. Premier essai Freeway (graine 98, 21 octobre 22 h 31) — échec G1, corrections, nouvel essai
Résultats conservés dans `craft_v6_out/v80/attempt1/`. G0 réussi (99,9 % des voitures vues à
≤ 4 px). Étiquettes de points justes (0,95 test, 0,91 entraînement ; aucune mort dans Freeway).
G1 Freeway échoué : positions imaginées à 16 pas justes à ≤ 4 px pour 66 % (seuil 80 %) ; aucun
contact voiture-poule enregistré (registre vide, a priori uniforme). G2 nul (0,7 % d'écarts).
Causes (défauts d'implémentation, pas des réglages) : (1) les objets étaient découpés sur
l'avant-plan entier au lieu de couleur par couleur comme prévu au §3 : une voiture sur la poule ne
faisait qu'un bloc, donc aucun contact ; (2) la vitesse était estimée sur un seul pas : une voiture
qui avance d'un pixel tous les deux pas lisait 0 ou 1 au lieu de 0,5. (3) Le score de Brier sur la
mort est déclaré « sans objet » quand le test ne contient aucune mort (il valait −2,85 × 10⁹).
Corrections : découpage couleur par couleur ; vitesse sur les 8 dernières positions observées ;
tests ajoutés. Seuils inchangés ; nouvel essai sur la même graine de développement 98.
Essai 2 (22 h 36), arrêté avant la collecte : le découpage couleur par couleur casse chaque voiture
en 3 à 5 morceaux (carrosserie, roues, vitres) : 47,5 objets par image, la limite de 48 atteinte en
permanence (G0 restait à 99,9 %, `attempt2/`). Correction retenue : blocs entiers (une voiture
reste un objet) et, quand un bloc mêle MES couleurs (connues par mes sondes) à d'autres, ma partie
est séparée du reste — le contact avec moi reste visible. Freeway : 12,8 objets par image (max 16).
Essai 3 (22 h 41, traces neuves, blocs + séparation de mes couleurs) : G0 99,9 %, 12,3 objets par
image. Contacts voiture-poule enfin enregistrés (« je suis repoussé » de 11 à 19 px vers le bas ;
aucune mort) ; vitesses fractionnaires justes. G1 échoué : positions à 16 pas justes à 73 %, score
pire que la climatologie (−0,45) ; G2 −0,02 (`attempt3/`). Causes : le suivi perd les voitures qui
reviennent par l'autre bord ; la théorie ne relie les points qu'aux contacts, alors que Freeway
donne le point en ATTEIGNANT le haut ; une couleur de la poule non retenue par mes sondes était
prise pour un objet qui me suit.
Essai 4 (22 octobre 0 h 30, mêmes traces, théorie seule) : notions générales ajoutées — ce qui
reste collé à moi est moi ; rebouclage par couleur ; zones qui donnent un point ou tuent (événement
rattaché à ma position quelques pas avant sa confirmation par le lecteur de compteurs) ;
replacement après un événement. Théorie apprise : 7 couleurs de voitures rebouclent ; zone de
points tout en haut (27 % par pas) ; replacement en bas après un point (47, 191). G1 Freeway
RÉUSSI : 85 % des positions à ≤ 4 px ; score +0,17 [0,04 ; 0,28] au-dessus de la climatologie
(`attempt4/`). Réserves : (1) le test avait été vu à l'essai 3 — à CONFIRMER sur la graine 99
(traces et test neufs) ; (2) le critère « P(je suis repoussé sous 16) » n'est pas encore codé et
le rapport à l'extrapolation est vide (médianes nulles) ; (3) la théorie ne dévie jamais de
« monter » (G2 0,0 ; Freeway G2 est seulement rapporté).
