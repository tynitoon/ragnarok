# Premier fit sur les images Atari : résultat négatif

10 septembre 2026. L’exécution est complète et sa validité a été revue
indépendamment. Le critère scientifique n’est pas atteint. Cet essai ajuste
un prédicteur de zones de contraste ; il n’entraîne pas une politique qui joue.
Il ne démontre ni maîtrise d’un jeu ni apprentissage accéléré du suivant.

Le réseau reçoit deux images passées, sans bouton, jeu ou coordonnées. Sa cible
est l’union des pixels qui diffèrent entre nos commandes et NOOP sur huit
images futures. Ce masque n’est pas une annotation d’objet physique. Les douze
situations viennent de six jeux et sont toutes des données de développement.

## Contrat et exécution

Sources 842fd9b puis 4138fb8, protocole `ARC4_RESPONSE_ENVELOPE_FIT_PROTOCOL.json`,
préflight cab625d. FCN 10 177 paramètres, même initialisation dans les deux bras,
256 updates visuelles et 256 sur entrée constante, alternées sur le même lot.
CPU un thread, zéro GPU et zéro nouvelle image émulée. Temps de travail :
738,146 s, sous le plafond de 1 200 s. Aucun retry ou prolongement.

Les 512 mises à jour Atari se comptent séparément des 1 280 mises à jour des
cinq fits synthétiques précédents, eux aussi négatifs à leur critère global.

La revue a vérifié 90 sources, 15 preuves et les 2 049 événements du journal.
Les 48 lignes de métriques — douze ancres, quatre bras ou témoins — concordent
au recalcul indépendant, avec un écart maximal de 1,11e-16. Les poids sont finis,
aucun step n’est incertain. **PASS de validité de mesure n’est pas PASS du fit.**

## Résultat au seuil fixé de 0,5

| Clause fixée avant le fit | Mesure | Résultat |
|---|---:|---|
| IoU moyen par jeu ≥ 0,25 | 0,047393 | Non atteint |
| Avance sur le meilleur témoin ≥ 0,10 | 0,042491 | Non atteint |
| Faux positifs ≤ 0,1 % sur chacune des deux cibles vides | 0,5655 % ; 0,1607 % | Non atteint |

| Jeu, moyenne de ses deux ancres | IoU avant ajustement | IoU visuel final |
|---|---:|---:|
| Pong | 0,00655 | 0,00980 |
| Breakout | 0,00786 | 0,02185 |
| Freeway | 0,00378 | 0,03711 |
| Ms. Pac-Man | 0,00436 | 0,13485 |
| Asteroids | 0,00196 | 0,03336 |

Seaquest n’entre pas dans cette moyenne : ses deux cibles observées sont vides,
avec respectivement 190 et 54 faux positifs sur 33 600 pixels. Cela ne dit pas
que Seaquest est incontrôlable, seulement que ces deux fenêtres n’ont révélé
aucun contraste dans le protocole de collecte.

Le témoin initial classe tous les pixels positifs. L’entrée constante entraînée
donne partout 0,416665524, près de son optimum BCE analytique 5/12 ; au seuil
fixé, son masque est vide, comme le témoin sortie zéro. Leurs IoU sont nulles
sur les cibles non vides. Leur AP n’est pas nulle : avec des scores tous liés,
elle vaut la prévalence du masque. L’AP visuelle moyenne par jeu est 0,15666 ;
elle ne remplace aucune clause du critère.

La BCE du bras visuel descend de 0,71929 à 0,28637 dans le journal. Ces valeurs
sont évaluées **avant** les mises à jour 1 et 256, et ne sont pas une mesure
supplémentaire de perte aux poids finaux. La courbe descend encore : ni
convergence, ni minimum local, ni limite de capacité ne sont établis.

## Lecture des images et suites

Les six planches de `v70_response_envelope_fit/errors_v0/` montrent les douze
cas, sans sélection : deux passés, cible, masque final et erreurs. Le réseau
couvre une partie des zones attendues mais produit de nombreux faux positifs,
notamment près des bordures et d’autres éléments graphiques. C’est une lecture
descriptive après le résultat, pas une nouvelle mesure confirmatoire.

La validation sur de nouvelles trajectoires, écrite avant ce résultat dans
`ARC4_RESPONSE_ENVELOPE_VALIDATION_DRAFT.md`, était conditionnée à la réussite
du fit. Elle n’est donc pas lancée. Les jeux réservés ne sont pas ouverts.

Un diagnostic externe séparé est déclaré dans
`ARC4_RESPONSE_ENVELOPE_EQUIVALENCE_DRAFT.md` : rechercher des entrées locales
exactement identiques portant des cibles opposées. Il borne certaines
performances possibles de l’opérateur local ; il n’optimise aucun réseau et
ne peut à lui seul désigner la cause de l’échec. La piste d’apprentissage
suivante exige une hypothèse distincte et une décision technique explicite.

## Pièces conservées

- Résultat et poids : commits 3473f13 et 963945e ; dossier
  `craft_v6_out/v70_response_envelope_fit/`.
- `fit_v0.json` : SHA256
  `9472d1153ea590b50fb3bd0a8f37baf5a989392604ade6bde60501a5155bad4a`.
- Revue 724ab24, `fit_review_v0.json` : SHA256
  `3df95a49c616d980c940b9ae20260a8e5c912560eecf07c6e0a8ad7f122d0467`.
- Courbes et scores, source 6e5099d, figure 3d85bc6 : `fit_v0.png`.
- Planches d’erreurs, source 7fabd1f, sorties 3fc15b8 : `errors_v0/atlas.json`.

Aucun artefact ni seuil antérieur modifié ; aucune nouvelle conclusion sur
le transfert entre jeux ne découle de ces visualisations.
