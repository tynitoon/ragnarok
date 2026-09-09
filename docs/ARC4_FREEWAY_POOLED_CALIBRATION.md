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
compte aussi 2048 fenêtres et 8323 frames, toutes incluses. Le résultat réservé
passe les douze types sans changement : joueur présence100%, boîtes2044/2048
(99,8047 %), deuxième poulet exact. Voitures : boîtes100% à un pixel, présence
car2 toujours2031/2048, autres100%. Leurs trajectoires déterministes répétées
ne constituent pas des réplications indépendantes entre seeds.

## Chaîne CPU complète

16 workers ×2048 décisions : NumPy878,38décisions/s, backend de calcul RGB
compilé1430,58décisions/s. Les16rapports sont strictement identiques, dont
comptes, perception et cibles C ; reçu `cpu_native_equivalence.json`. Émulation,
IPC RGB, extraction, juge à chaque frame et branches sont compris. Démarrage
séparé5,5335s pour le natif, calcul22,9054s. Charge ARC3/hôte variable entre
exécutions : pas de facteur de gain contrôlé. Minimum technique1000 dépassé,
cible provisoire2000 toujours non atteinte. Aucun verdict de gate.
