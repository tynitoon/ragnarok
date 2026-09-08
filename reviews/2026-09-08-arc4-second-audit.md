# ARC 4 — seconde revue du brouillon, 8 septembre 2026

Reprise autorisée par Jérémie après la pause de la session Claude. Périmètre : lecture intégrale de
`ARC4_VISION.md`, puis corrections documentaires bornées. Aucun gate, entraînement GPU ou test du
prototype n'a été exécuté dans cette revue. Le document reste un brouillon. La première revue (§10)
est préservée ; les décisions et blocages actuels sont récapitulés dans sa nouvelle §11.

L'objectif reste de démontrer qu'un agent apprend grâce à ses acquis et apprend ensuite plus vite.
Les choix du premier audit sont conservés : perception apprise par expériences propres, aucune
référence classique comme professeur, RAM réservée au juge, gates peu coûteux et résultats négatifs
conservés. Les corrections suivantes ne constituent pas un nouveau protocole confirmatoire.

## Cinq défauts matériels

1. **Comptabilité incompatible avec la cadence des branches — §§2.1, 2.2, 2.5, 5.** Une paire C de
   deux branches de 4 frames avec deux restaurations facturées 60 coûte 128 frames expérimentales.
   Toutes les 16 frames ordinaires, elle représente 128/144, soit 88,9 % du total ; le plafond est
   pourtant de 30 %, avec une allocation prévue de 20 % puis 5 %. La nouvelle convention sépare
   frames ordinaires, frames expérimentales, charges et agent-steps. C seul nécessite au moins
   512 frames ordinaires à 20 %, puis 2432 à 5 %. E1–E4 partagent cette allocation et ralentissent donc
   encore C. L'inventaire complet proposé pour E1 Atari coûte au moins 39 168 frames facturées avec
   deux restaurations par essai, avant les éventuels chargements d'ancres : il dépasse l'entrée de
   20k. La cadence toutes les 16 frames et l'estimation E1 de 9k sont retirées. La couverture réelle
   des essais et la décroissance du budget restent à décider avant gel, sans essai gratuit.

2. **La reproductibilité contrefactuelle était supposée — §§2.1, 2.5.** Le code officiel ALE précise
   que `cloneState()` exclut le RNG par défaut et que sa capture est explicite. Le contrat exige donc
   l'état complet, le RNG et les buffers/compteurs des wrappers ; le compteur global de frames
   facturées ne doit jamais reculer. Avant GPU, deux branches de même action issues du même état
   doivent rendre des pixels identiques, bruts et max-poolés. La convention sticky du laboratoire
   doit être fixée explicitement ; l'évaluation conserve p = 0,25. Cela valide la reproductibilité,
   pas à lui seul l'interprétation de chaque différence comme contrôle d'un objet.
   [Source primaire ALE consultée le 8 septembre 2026](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/master/src/ale/ale_interface.hpp).

3. **La référence ne certifie pas la vérité terrain — §§2.1, 4.1, 5.** Un extracteur par fond modal
   et composantes connexes peut échouer alors que les coordonnées RAM sont correctes. La clause
   « recall < 0,9 implique harness RAM erroné » est remplacée par deux vérifications séparées :
   calibration du juge sur fixtures écran/RAM, puis mesure de la référence contre ces labels.
   Coordonnées, visibilité, HUD, flicker, synchronisation et dénominateur des objets doivent être
   précisés avant les résultats. Le seuil de référence proposé de 0,9 est conservé ; un échec bloque
   ce gate fondé sur la référence et demande un diagnostic, sans désigner automatiquement le juge.

4. **Une avance fixe satisfaisait encore « learns faster » — §§0, 1.2, 4.3, 4.4, 9.** Contre-exemple :
   X gelé à HN = 0,30 dès l'entrée, fresh n'atteignant C2 qu'à 4 M. X domine plusieurs fractions de
   franchissement sans apprendre sur le nouveau jeu. La fraction de checkpoints est donc conservée
   comme diagnostic de compétence atteinte plus tôt ; elle ne suffit pas à établir une adaptation
   accélérée. La manière de séparer adaptation et avance reste ouverte avant gel confirmatoire.
   Aucune nouvelle statistique ni condition n'est ajoutée. K*, entraîné directement sur les jeux de
   développement, devient une référence favorable : un entraînement fini n'en fait ni une borne
   supérieure universelle ni une preuve d'impossibilité en cas d'échec. Son entraînement borné, sa
   gestion du ledger à l'entrée et son rôle décisionnel restent à figer avant son propre gate.
   FRESH-COST mesure un goulot possible ; une perception peu coûteuse ne prouve pas que la politique
   et les connaissances de contact sont elles aussi peu coûteuses à apprendre.

5. **Les jeux de gate et de confirmation étaient confondus — §§3.1, 4.4, 4.6.** Les jeux utilisés pour
   entraîner K*, déboguer ou choisir une correction sont consommés pour la confirmation du même
   stade. Le texte distingue désormais entraînement, développement/gates et confirmation intacte.
   Les candidats Atari Tennis, Krull, Enduro, Boxing, Frostbite, Riverraid n'en laissent que trois
   après les ouvertures proposées par gate 0, alors que six étaient annoncés. Le registre est déclaré
   incomplet ; aucun jeu ni effectif manquant n'est inventé. Les allocations NES/16-bit restent elles
   aussi à résoudre avant les gels correspondants.

## Prochain jalon et limites

Travail CPU utile, dans cet ordre : comptabilité et admission des expériences ; pool d'émulation et
restaurations exactes ; calibration indépendante du juge ; référence et débit complet ; registre,
commandes et critères exécutables de gate 0. Geler ensuite ce seul gate, avec son plafond proposé de
12 GPU-h et ses sorties/reprises prévues. Le gel de gate 0 ne gèle pas les décisions confirmatoires
encore ouvertes. Les corrections du brouillon précèdent toute observation ARC 4.

Le U-Net, le regroupement par mouvement et le lecteur odométrique restent des hypothèses
expérimentales. Cette revue identifie des contradictions de protocole et de conclusion ; elle ne
prétend pas démontrer leur réussite ou leur impossibilité. Les estimations de temps et de stockage
doivent encore être mesurées sur l'environnement local.
