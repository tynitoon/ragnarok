# Chaîne CPU Pong — diagnostics de développement

**Erratum comptable du 9 septembre :** les premiers essais omettent 60 frames
d'auto-détection au chargement de chaque processus. Le coût de sélection de mode
des resets naturels était aussi supposé nul sans journal exhaustif. Voir
[audit des resets](ARC4_RESET_AUDIT.md). Les débits ordinaires chronométrés et
l'équivalence des sorties restent mesurés ; ces anciens comptes ne peuvent pas
servir de preuve d'un plafond strict. Les résultats historiques sont préservés.

Le benchmark traite une vraie requête/réponse avec image RGB par décision,
répète l'action quatre frames, décode chaque frame côté juge et évalue les
instances au dernier état brut. Il crée une paire de branches toutes les
128 décisions, avec admission et coût des deux restaurations. Les cibles C
proviennent de la différence des images, sans RAM.

## Premier essai conservé, à corriger

`cpu_chain_2_workers.json` : 454 décisions/s ; `cpu_chain_16_workers.json` :
1 585 décisions/s, 512 décisions par processus. Initialisation séparée du
chronométrage ; calcul ARC 3 actif. Ces données ne sont pas un verdict de gate.

La revue a identifié deux défauts de portée avant adoption :

1. Le fond de référence était réinitialisé à la frontière d'épisode indiquée
   par l'émulateur. Cette information devait rester dans le juge ; le test de
   décodeur RAM vide ne couvrait pas ce chemin. La référence doit conserver son
   histoire de pixels sans ce reset oracle, ou être explicitement un contrôle
   bénéficiant d'un reset oracle.
2. Les observations ordinaires étaient brutes, mais `branch_pair` renvoyait des
   images max-poolées. Les masques C étaient bien construits depuis les pixels,
   mais sur une représentation différente. Le prochain essai brut devra utiliser
   les images brutes également pour les branches.

Les rapports initiaux, leurs sources et leurs chronométrages restent conservés.
L'intitulé brut du premier JSON ne décrit donc pas entièrement ses branches :
ce défaut est explicité ici, sans modifier rétroactivement le résultat.

La variante max-poolée complète reste une étape ultérieure, avec calibration
temporelle dédiée. La valeur proposée de 2 000 décisions/s pour le gate ne peut
pas encore être jugée depuis ce benchmark sur Pong brut seul.

## Version corrigée

La référence conserve maintenant uniquement son histoire de pixels, y compris
au changement d'épisode. La frontière d'épisode demeure dans le juge pour éviter
les faux mouvements. Un moteur dérivé sélectionne le brut pour toutes les
observations et toutes les branches ; le moteur original et le décodeur scellé
ne sont pas modifiés. La revue indépendante confirme la clôture des deux défauts.

18 tests passent ; ils comprennent un décodeur RAM volontairement vide qui
change les scores mais pas les pixels, instances ou masques C, ainsi qu'une
fixture où les images brutes et max-poolées diffèrent réellement.

| Essai corrigé | Décisions ordinaires par processus | Débit ordinaire | Initialisation |
|---|---:|---:|---:|
| 16 processus, court | 512 | 1 638 décisions/s | 4,23 s |
| 16 processus, prolongé | 2 048 | 1 200 décisions/s | 19,64 s |

Fichiers : `cpu_chain_16_workers_corrected.json` et `cpu_chain_16_workers_long.json`.
Le second couvre aussi des resets réels en cours de partie. Le calcul ARC 3 est
actif ; ces mesures sont sensibles à la charge simultanée et ne constituent pas
une estimation de débit au repos. Le numérateur publié est explicitement celui
des décisions ordinaires ; le débit incluant les décisions expérimentales est
fourni séparément. Tous les coûts de branches et restaurations restent comptés.

Le diagnostic de référence sans reset oracle sur 17012 donne un rappel mobile
de 99,691 %, une précision de 99,051 % et un rappel balle de 99,298 % :
`reference_validation_no_oracle_reset.json`. Les anciens résultats avec reset
oracle sont conservés comme contrôles et ne sont pas réinterprétés rétroactivement.

Suite : vérifier séparément la convention max-poolée, puis mesurer la chaîne
correspondante. La cible proposée de 2 000 décisions/s reste non atteinte par
ces essais bruts ; l'extension aux autres jeux et les primitives restent à faire.

## Chaîne max-poolée calibrée

La validation réservée 17022 passe après scellement : voir
`docs/ARC4_PONG_POOLED_CALIBRATION.md`. Le benchmark impose un résultat de
calibration passé dont les empreintes correspondent au décodeur et au protocole.
Le fond ne reçoit toujours aucune frontière d'épisode ni RAM.

`craft_v6_out/v70_pong_pooled/cpu_chain_16_workers.json` : 16 processus,
2 048 décisions ordinaires chacun, 33,59 s de travail et 5,67 s d'initialisation.
Débit ordinaire **975,52 décisions/s**, ARC 3 actif. Les branches, restaurations,
échanges RGB, référence et juge sont inclus. Cette mesure est sous le minimum
d'ingénierie provisoire de 1 000 et sous la cible de 2 000 ; aucun seuil n'est
abaissé. Le profilage attribue une part importante au comptage du fond modal
(boucles par couleur et recherche des couleurs à chaque image). Prochaine action :
optimisation exacte, avec équivalence des sorties et nouveau chronométrage.

L'optimisation remplace les balayages par couleur par un incrément du seul
compteur concerné à chaque pixel, avec index compact des couleurs observées.
Le fond garde exactement sa définition et ses égalités stables. La palette est
apprise depuis les pixels ; les requêtes de branches restent sans mutation.
25 tests passent en 4,57 s, dont une comparaison avec des historiques Python
indépendants, croissance de palette, grande indexation et échecs sans mutation.
La revue indépendante ne relève aucun défaut matériel.

Résultat `cpu_chain_16_workers_scatter.json` : **1 976,22 décisions/s**, 16,58 s
de travail et 3,03 s d'initialisation, mêmes 16 × 2 048 décisions. Tous les
rapports des processus sont exactement égaux à la baseline : scores, coûts,
branches et empreintes des cibles C. Preuve conservée dans
`cpu_scatter_equivalence.json`. Le minimum provisoire de 1 000 est dépassé,
la cible de 2 000 reste légèrement non atteinte. Les deux chronométrages sont
séquentiels sous charge variable ARC 3 : le rapport de 2,03 n'est pas une
estimation contrôlée du gain. La chaîne multijeu et les primitives restent à faire.

## Intégration de la convention à construction fixe

Après l'erratum des resets, le moteur froid est intégré explicitement avec
`--reset-convention cold`. Le hook du juge est réinstallé à chaque objet ALE
neuf. Les tests suivent sa paire RAM dans les frames terminales et les nouveaux
épisodes ; un juge volontairement corrompu ne change pas les pixels, instances
ou cibles C. Les huit tests du pool de référence passent en 7,24 s.

`craft_v6_out/v70_pong_pooled/cpu_chain_16_workers_cold.json` : 16 × 2 048 décisions,
**1 592,62 décisions/s**, 20,57 s de travail et 3,85 s d'initialisation. Les 48
constructions (trois par processus) sont entièrement facturées ; total 170 162
frames, branches et restaurations comprises. Le numérateur chronométré reste
celui des décisions ordinaires ; les coûts des 32 reconstructions pendant le
travail sont inclus. ARC 3 tourne toujours. Le minimum provisoire de 1 000 est
dépassé, la cible de 2 000 reste non atteinte. Aucun verdict de gate.
