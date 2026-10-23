# v83 — résultat d'A1 (G-place) : KC-place

Verdict officiel : `craft_v6_out/v83/room_verdict.json` (vérificateur corrigé, note 21) ; première
notation gardée : `room_verdict_first_pass.json` (Interruption). Revue contradictoire du verdict :
workflow `wf_420b5f0a-bc1` (23 octobre). Sens utilisés : hausses du score et vies (décision de Jérémie du
22 octobre).

## Résultat

| Lecture | Jeux à place (dév.) | σ̂² | se_proj |
|---|---|---|---|
| Officielle (vérificateur corrigé) | 6/35 | 1,165 | 0,480 |
| Première notation (label « Interruption ») | 6/35 | 1,165 | 0,480 |
| Sans Q\*bert | 5/35 | 1,389 | 0,568 |
| σ̂² poolé avec breakout et ms_pacman | 6/35 | 1,002 | 0,450 |
| (d) plus strict (sans bank_heist ni jamesbond) | 4/35 | 0,227 | 0,324 |

Seuils figés : au moins 13 jeux et se_proj ≤ 0,23. Jeux à place : alien, amidar, bank_heist, ice_hockey,
jamesbond, qbert ; calibration : breakout, ms_pacman. Le verdict tient sur le compte seul (sans
bank_heist, σ̂² tomberait à 0,473 ; avec 13 jeux, se_proj serait 0,228).

## Phrase figée et ce que montre la carte des coûts

Phrase figée (gardée telle quelle) : « avec retours en arrière et les sens, l'entraînement neuf de cet
explorateur est trop bon marché, saturé ou bruité pour qu'une montée 2× plus rapide soit mesurable sur le
lot scellé ».

Ce que montre la carte des coûts : pas « trop bon marché » (aucun jeu écarté par (d)). Surtout trop lent
pour cet explorateur : sur 12 jeux de développement, une graine n'atteint pas M0 ; 33 des 58 runs écartés
améliorent encore leur meilleur après 1,5 M ; atlantis, bowling, centipede et krull restent sous la barre
B sur les deux graines. « Bruité » est réel mais porté surtout par bank_heist (4,62 sur 6,99 ; parts de
sondes 0,004 contre 0,75). « Saturé » au sens d'un plafond du jeu ne vaut que pour fishing_derby (M3 = 270
> 99) et, en calibration, pour pong (62,1 > 21) et freeway (106 ; meilleurs runs 21 et 23). Portée : la
mesure vaut pour un explorateur qui n'archive jamais un état après une vie perdue, comparé à des barres
jouées sur toutes les vies, souvent par un hasard que l'explorateur ne joue pas (B / hasard répété ≥ 2,3
sur 21 jeux, aucun à place ; constat fait après les résultats). zaxxon est à la barre, pas dessous
(B = 0, 0/0).

## Ce qui est établi, ce qui ne l'est pas

Établi : cet explorateur générique (cases moi × vies × score, déroulés au hasard tenus, sondes de
« moi »), avec retours en arrière et les sens, ne monte assez en 2 M d'images que sur quelques jeux
Atari. Non établi : qu'apprendre plus vite soit impossible ; rien n'a été mesuré sur les promesses ni sur
le transfert, et les lots scellés n'ont pas été ouverts.

## Leçons pour la suite (classées, revue `wf_420b5f0a-bc1`)

1. Obtenir d'abord un explorateur neuf qui monte (porte « l'explorateur neuf atteint les paliers sur la
   plupart des jeux ») avant tout bras de transfert : (b) seul écarte les 29 jeux sans place ; 4 jeux
   restent sous la barre ; 33 runs sur 58 progressaient encore après 1,5 M.
2. Construire la barre avec le même processus de boutons que l'explorateur, ou laisser l'explorateur
   jouer génériquement les modes simples de la barre (changement à chaque pas, bouton tenu longtemps) ;
   à déclarer avant tout résultat.
3. Plafonner le budget des sondes et mesurer la perte de « moi » pendant l'exploration (sondes : 26,8 %
   des images, ≥ 64 % sur 7 jeux ; krull n'a fait que 208/159 déroulés).
4. Vérifier avant de figer que chaque palier est atteignable, et mettre barre et explorateur sous la même
   règle de vies (fishing_derby, pong, freeway ; 91 % des déroulés d'atlantis s'arrêtent sur une vie).
5. Donner à la case un axe de progrès qui ne dépende ni de « moi » ni du score (gopher 9131 et
   name_this_game 9132 : ~4 100 déroulés sans hausse ; zaxxon 0/0).
6. Rendre le terme de bruit robuste (plus de graines ou estimateur robuste, déclaré d'avance).
7. Procédure : committer les négatifs gardés, nommer le code de chaque relance, tamponner la provenance au
   lancement et à la fin, garder l'historique des arrêts du superviseur avec les sorties, livrer chaque
   rapport promis ou le déclarer reporté, ne jamais renoter un verdict sans le dire dans le label.
8. Lots : les 24 jeux scellés et les 9 du second tour restent fermés et utilisables sous un nouveau
   protocole ; tout changement de l'explorateur = nouvelle version et nouvelle porte A1.
