# Freeway — calibration des deux images sources

Protocole/candidat fixé `ed7fd7a` avant 17221. Le juge brut V2 reste inchangé.
Chaque identité reçoit l'enveloppe de ses boîtes visibles. Le checker RGB retient
seulement ses pixels qui contribuent au maximum des deux sources, sans RAM.
Il décrit ces contributions ; l'enveloppe peut inclure du vide et certaines
identités peuvent être indécidables depuis le seul maximum RGB.

Baseline préservée `7719d3f`, sans correction : douze types passent sur 2048
fenêtres. Joueur : présence2044/2048, boîtes2035/2044 (99,5597 %). Car2 :
présence2031/2048 (99,1699 %), marge étroite ; tous ses défauts restent inclus.
Les cas car2 inspectés correspondent à une colonne RAM x159 alors que les deux
RGB montrent seulement le fond : résidu de phase du brut, pas un décalage du tap.

Revue indépendante Ohm close sans défaut matériel : dix empreintes concordantes,
2047 répétitions complètes puis trois frames, 8191+132=8323 frames facturées.
Sceau commité avec succès `0c96911` **avant** capture réservée 17222. Cette capture
compte aussi 2048 fenêtres et 8323 frames, toutes incluses. Résultat réservé
en cours d'évaluation à ce jalon ; aucun critère ou candidat n'est modifié.
