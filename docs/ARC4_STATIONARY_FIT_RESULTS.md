# Contrôle stationnaire : représentation possible, ajustement insuffisant

10 septembre 2026. La construction et son certificat ont réussi. Le fit séparé
depuis une initialisation aléatoire est complet, mais manque les trois clauses
du critère. Sa validité instrumentale est confirmée par une revue indépendante.

La tâche conserve douze cas simples : dix images avec des pixels dont la couleur
change sous notre commande, et deux images sans réponse. Les labels viennent
uniquement de la comparaison des branches observées. Le réseau reçoit les deux
images passées. Les poids du certificat ne sont jamais chargés par l’apprenant.

## Deux résultats différents

La référence construite dans le juge utilise exactement le même réseau de
10177 paramètres. L’identité SiLU(t)−SiLU(−t)=t permet une solution centrale
dont les logits valent±8. Le calcul unique confirme IoU/AP1 et zéro faux positif,
avec une BCE0,000335574. Ce résultat prouve la représentabilité sur cette
famille ; il ne prouve pas que l’apprentissage retrouve ces poids. Cette BCE
est une valeur atteignable, pas une borne inférieure du problème séparable.

L’apprenant séparé conserve la recette du fit Atari : initialisation764,
Adam3e-4, clip1, BCE équilibrée,256 mises à jour par bras sur les douze cas.
Le bras à entrée nulle reçoit une copie exacte des poids aléatoires initiaux.
Le fit dure744,030s, sans erreur,512 updates exactement comptées,512 forwards
d’apprentissage et3 d’évaluation. Aucun nouvel écran ni calcul GPU.

| Clause fixée avant l’essai | Mesure | Verdict |
|---|---:|---|
| IoU visuelle moyenne≥0,25 | 0,00492999 | Non atteint |
| Avance≥0,10 sur le meilleur témoin | 0,00240916 | Non atteint |
| FP≤0,1% sur chacun des deux vides | 0,23214% ;0,47024% | Non atteint |

Les deux vides ont78 et158 faux pixels, pour un maximum admissible de33 chacun.
L’AP moyenne est0,203889. La BCE visuelle finale est0,558961, tandis que la
constante entraînée approche son optimum de fréquence5/12. L’initial prédit
tout positif, avec IoU0,00252083. La figure conserve les dix cas non vides et
les deux vides séparés ; aucune ancre n’a été retirée.

Le cas à un pixel illustre la limite : le pixel correct a le score le plus élevé,
donc AP1, mais370 autres pixels franchissent aussi le seuil fixé0,5. Le seuil
reste inchangé ; une recherche après résultat ne transformerait pas ce FAIL
en réussite du protocole.

## Revue et limites

Sources du fit0e5ede9, préflightc04a9c5, lancement WMIee6f32c, résultat19f5ef4.
Revue finale4c612de :106 sources,26 preuves,48 lignes IoU/AP/FP, trois BCE et
2055 événements vérifiés. Les poids initiaux correspondent bit à bit à ceux
de la recette aléatoire archivée. Aucun forward supplémentaire dans la revue.

La première revue s’est arrêtée avant de décoder les tableaux : son contrôle
Git utilisait des chemins Windows absolus, qui n’appliquaient pas la même
conversion de fins de ligne. Ses traces restent en place. Le réviseur a été
corrigé pour utiliser des chemins relatifs, puis a produit un nouveau reçu
v1 sans déplacer ou remplacer le premier. Le temps cumulé12,41s inclut cette
première tentative ; aucun artefact du fit n’avait changé.

Le résultat signifie que cette recette, sur ce lot et dans ce budget, ne
s’ajuste pas assez à une tâche dont une solution de l’architecture est connue.
Il ne prouve ni minimum local, ni convergence, ni impossibilité d’apprendre.
La baisse de BCE et quelques bonnes positions ne suffisent pas au critère.

La validation conditionnelle préparée AVANT le fit n’est pas exécutée : sa
condition est fausse. L’instrument préparédef8908 conserve ses SHA d’entrée
àNone et ne peut pas lancer de génération. Les essais Atari et le raccord
P→m→C/ME restent suspendus. Aucun jeu maîtrisé ni transfert accéléré démontré.

Preuves :

- `craft_v6_out/v70_stationary_fit/fit_v0.json`,
  SHA8ca241233e8053cedd6149b1d3992dc7cbc944c4442ce298b067dafc03719c35.
- `craft_v6_out/v70_stationary_fit/fit_review_v1.json`,
  SHAc2fe9c270afa098f43c403b3f4818460bd9540993794049cd7df04f1151daa98.
- `craft_v6_out/v70_stationary_fit/fit_v0.png`, figure générée une fois à partir
  des JSON revus, vue et vérifiée, sans nouvelle inférence.
