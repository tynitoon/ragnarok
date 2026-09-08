# Breakout — calibration temporelle CPU

Protocole fixé dans `ARC4_BREAKOUT_POOLED_PROTOCOL.json`, commit `10d88cc`, avant
la capture de développement 17121. Les 2048 observations sont les maxima des
deux dernières images réelles du moteur, répétition quatre, avec leur RAM
alignée. Le juge brut scellé `6e48ef4` reste inchangé. Aucun entraînement ni gate.

## Développement et revue avant validation

La baseline v0 passe sans correction : raquette 2029/2048 boîtes conformes à
un pixel (99,0723 %), balle 2035/2048 accords de présence (99,3652 %) et
1589/1597 boîtes conformes (99,4991 %). Les six types de rangées passent aussi.
La marge de la raquette est étroite ; les critères ne seront pas assouplis.

Les dix constructions froides coûtent 1320 frames, les actions 8179, total
9499. Toutes les fenêtres, y compris les six répétitions terminales tronquées,
restent dans les dénominateurs. Les neuf reprises sont intégralement comptées.

Revue indépendante Ohm terminée le 9 septembre 2026 avant toute ouverture de
17122 : alignement du tap après chaque nouvelle instance, maximum RGB effectif,
matching donnant priorité au nombre de boîtes conformes, dénominateurs et dix
empreintes de sources vérifiés. Aucun défaut matériel bloquant trouvé.

## Limite d'identifiabilité

Le checker indépendant décrit les contributions des **deux images sources**.
Il n'est pas une vérité sémantique univoque du seul maximum RGB. Une balle rouge
compacte dans A recouverte par une raquette du même rouge dans B donne le même
maximum que (B,B), mais les provenances étiquetées diffèrent. Les égalités RGB
conservent les deux contributions conformément au protocole préalable.

La calibration valide donc la concordance de deux instruments temporels,
pas la possibilité d'apprendre toutes ces identités depuis l'image fusionnée.
Toute mesure ultérieure de perception doit conserver cette portée, sans
présenter une erreur d'occlusion indécidable comme nécessairement apprenable.
Les approximations du juge brut aux contacts, dans les rangées rouges et en
bas de l'écran restent présentes et pénalisées. Aucun oracle n'entre dans
l'extracteur de référence.

## Validation réservée

Candidat conservé dans `eb21d20`, sceau `9dcb1b8` avant ouverture de 17122.
Les dix sources sont restées identiques. Sur 2048 fenêtres, les huit types
passent : raquette 2036/2048 boîtes conformes (99,4141 %), balle 2043/2048
accords de présence (99,7559 %) et 1578/1584 boîtes conformes (99,6212 %).
Rangées 0–4 exactes ; rangée 5, 3138/3141 boîtes conformes (99,9045 %) et
2045/2048 accords de présence. Onze constructions, 9627 frames facturées.

Résultat intégral : `craft_v6_out/v70_breakout_pooled/validation_result.json`.
Les erreurs résiduelles et la limite d'identifiabilité ci-dessus restent valables.
