# Délai de divergence et reconnaissance de soi

La vision §2.2 propose d'utiliser une divergence dès la première image pour
distinguer l'objet directement commandé de ses effets ultérieurs. Cette règle
doit être considérée comme une hypothèse instrumentale, pas comme une propriété
générale des jeux. Aucun sélecteur ME appris ni résultat de gate n'est disponible.

Deux témoins RGB synthétiques exposent la limite sans consulter de jeu réel.
Le simulateur contient deux objets persistants A et B. À chaque image, la touche
déplace A, puis la nouvelle position de A fait immédiatement bouger B. Les deux
branches action/NOOP divergent alors dès leur première image, pour A comme pour
B. B est un effet indirect selon les équations du simulateur ; aucun modèle ne
reçoit cette annotation. Le simple numéro de l'image ne sépare pas ces causes.

Dans le second témoin, la touche précédente est maintenue lors de la première
image, puis la nouvelle touche s'applique lors de la seconde. A, pourtant
directement commandé, ne diverge qu'à la deuxième image. Ce cas illustre une
possibilité admise par les touches collantes ; il ne mesure pas sa fréquence
dans les données Atari. Les deux branches utilisent le même calendrier collant.

Les régions connues du simulateur servent seulement à vérifier les assertions
du test, jamais à produire des entrées, catégories ou labels du parser. Ces tests
ne concluent pas que l'heuristique complète ME échouera : sa persistance, son
historique de contrôlabilité et ses autres critères peuvent apporter de
l'information. Ils interdisent seulement de traiter « première image » comme
une preuve suffisante de commande directe, ou « deuxième image » comme une
preuve suffisante d'effet indirect. Un sélecteur futur devra conserver
l'incertitude et être évalué selon des critères fixés avant ses résultats.

Les sources de mesure de divergence et les protocoles existants restent
inchangés. Ce constat complète les problèmes distincts d'amorçage C/effets/ME
et de supervision P, sans ajouter de nouvelle règle d'exclusion du gate.
