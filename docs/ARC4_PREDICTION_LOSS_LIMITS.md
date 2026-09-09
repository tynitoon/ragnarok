# Limite constructive de la prédiction par déplacement et perte L1

Diagnostic synthétique avant entraînement Atari. Il concerne la perte P déjà
implémentée, pas les résultats d'un entraînement. Le parser n'est pas appelé et
aucun poids n'est optimisé. Les déplacements ci-dessous sont les variables d'un
test mathématique de la fonction de perte ; aucune direction vraie n'est fournie
comme entrée ou cible au modèle.

Le même carré de2×2pixels donne une source poolée aux positions4/5 et une cible
aux positions8/9 après quatre images. La translation correcte est donc de quatre
pixels. Dans la famille des translations uniformes, les déplacements−0,5,
−0,25,+0,25,+0,5 donnent tous la même perte L1 moyenne0,046875 et un gradient nul
par rapport à la translation. La translation4 donne une perte nulle. Tant que
les supports du carré reconstruit et du carré futur ne se recouvrent pas, cette
perte ne fournit pas de direction locale dans cette famille de mouvements.

Un second témoin déplace toute la source hors de l'écran. Le reconstructeur
produit alors une image noire et obtient une perte0,0234375, inférieure à celle
de l'image restée immobile, sans avoir expliqué le déplacement. Les trous sont
bien conservés dans le dénominateur comme prévu ; cela ne suffit pas à empêcher
cette solution imparfaite d'améliorer l'erreur par rapport au point de départ.

Ces exemples ne prouvent ni que tous les gradients du champ dense sont nuls, ni
que le parser convergera vers une image noire, ni que cet échec se produira à
une fréquence donnée dans Atari. Ils montrent que conserver l'ordre des images
résout une ambiguïté d'entrée mais ne garantit pas à lui seul un objectif facile
à apprendre. Il faut tester séparément la capacité à acquérir le mouvement sous
un protocole annoncé, avant de qualifier la vision d'apprise. Une éventuelle
modification de la perte sera une nouvelle candidate, avec les anciennes
sources et les anciens résultats conservés ; aucun critère de gate n'est changé.
