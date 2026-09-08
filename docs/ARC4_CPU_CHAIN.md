# Chaîne CPU Pong — diagnostics de développement

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
