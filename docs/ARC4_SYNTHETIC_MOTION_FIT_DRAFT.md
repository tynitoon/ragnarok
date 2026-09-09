# Ajustement synthétique de P : contrôle prospectif borné

Ce contrôle vise une question étroite : le modèle peut-il ajuster les deux
déplacements opposés du contre-exemple, avec P et un budget d'optimisation fixé ?
Les deux mêmes scènes sont utilisées pour l'ajustement et les mesures. Ce n'est
ni un test de généralisation, ni une validation Atari, ni un résultat de transfert.

Les deux architectures sont exécutées dans l'ordre poolée puis ordonnée, chacune
initialisée avec la graine764, déjà utilisée pour les tests de représentation.
Chaque mise à jour reçoit les deux exemples simultanément. Les entrées sont les
propres RGB synthétiques et une hypothèse d'effet UNKNOWN identique. Les cibles
sont uniquement leurs images futures, jamais les coordonnées du carré ou sa
direction comme annotation. L'image poolée garde les deux mêmes sources pour le
contrôle initial ; les cibles de ses deux exemples restent incompatibles.

Le budget est256mises à jour Adam par architecture, taux3e-4, betas0,9/0,999,
epsilon1e-8, sans décroissance des poids, norme des gradients plafonnée à1.
Seule la perte image L1 de P est optimisée. Aucun objectif m/C/G/I/S n'est ajouté
pour ce diagnostic d'ajustement ; cela ne définit pas la recette Atari future.
Les mesures sont faites après0/1/8/32/128/256mises à jour, sans arrêt anticipé
en fonction du résultat. La borne de travail CPU est300secondes, vérifiée avant
chaque mise à jour et à la fin ; les preuves de sources sont hors de ce délai.

Le contrôle poolé ne peut passer sous0,0234375 de L1 moyenne sur ces deux cibles,
par l'argument déjà vérifié. Le critère prospectif pour la variante ordonnée
est une erreur finale strictement inférieure à0,01171875, la moitié de cette
borne. Ce choix est annoncé avant toute optimisation. Une exécution numérique
complète est distinguée d'un critère d'ajustement atteint. Un échec ou un résultat
négatif est conservé ; aucun essai, graine ou durée de remplacement n'est prévu.

Le contrôle ne mesure pas la vitesse des architectures : leurs nombres de
paramètres diffèrent et ARC3 peut utiliser le GPU pendant ce travail CPU.
Aucune émulation ni archive réelle, aucun GPU, aucune sauvegarde de modèle
entraîné réutilisable. Les empreintes initiales/finales et les courbes sont
conservées ; ces poids de diagnostic ne seront transférés à aucun jeu.
Préflight indépendant committé obligatoire avant la première mise à jour.
