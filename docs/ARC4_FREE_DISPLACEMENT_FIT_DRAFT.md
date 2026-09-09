# Témoin prospectif : ajuster directement deux champs de déplacement

Les ajustements avec réseau, P natif puis P à trois échelles, sont terminés et
n'atteignent pas le seuil natif de 0,01171875. Leurs résultats restent inchangés.
La seconde revue scientifique recommande un seul témoin pour mieux situer la
difficulté : conserver les images, la reconstruction et la perte floutée, mais
optimiser deux champs denses indépendants sans réseau.

Ce témoin possède 1024 paramètres en float32 : deux composantes par pixel, pour
chacune des deux scènes 16 × 16. Tous commencent exactement à zéro, sans tirage
aléatoire ni direction privilégiée. Les images futures sont la seule supervision.
Aucun déplacement vrai, masque, catégorie ou identité n'entre dans l'objectif.
Les paramètres propres à chaque scène lui accordent une capacité de mémorisation
qui ne serait pas disponible pour un agent devant découvrir une scène nouvelle.

Un seul ajustement est prévu : 256 mises à jour Adam, lot de deux scènes,
β=(0,9 ; 0,999), ε=1e-8, aucun weight decay, norme des gradients bornée à 1.
Le pas est **0,05 pixel par coefficient Adam**, déclaré avant l'exécution. Les
coefficients sont ici directement des pixels de déplacement ; réutiliser le
pas 0,0003 des poids du réseau ne constituerait pas une comparaison équivalente.
0,05 donne une échelle nominale de vingt mises à jour par pixel, laissant place
à un déplacement de plusieurs pixels dans les 256 mises à jour. Ce calcul ne
garantit ni le mouvement réel des paramètres, ni une convergence. Aucun balayage
de pas, redémarrage, deuxième initialisation ou prolongation n'est prévu.

La source reste le maximum des deux écrans ordonnés déjà utilisés ; le futur
reste celui des deux scènes originales. Le reconstructeur par projection et
moyenne des collisions reste inchangé. L'objectif garde les trois pertes L1
de poids égaux aux largeurs 1, 3 et 7. Seul le critère **L1 RGB natif final
strictement inférieur à 0,01171875** peut donner un témoin positif. La perte
floutée et la séparation des deux reconstructions sont rapportées séparément.
Les champs, couvertures et images finales sont conservés comme tableaux de
diagnostic, pour permettre une vérification numérique sans réentraînement.

Un succès montrerait que ce couple perte/reconstructeur peut atteindre le
seuil avec ces champs libres, cette initialisation et cet optimiseur. Il
orienterait le diagnostic vers la paramétrisation ou l'optimisation du réseau,
sans isoler une cause unique. Un échec laisserait aussi en cause la perte,
le reconstructeur et l'optimisation des champs ; il ne prouverait pas une
impossibilité. Aucun résultat ne prouverait perception, généralisation, jeu
Atari, transfert ou apprentissage accéléré.

L'essai est CPU uniquement, un thread numérique, sans émulation ni GPU. Borne
coopérative de 300 secondes de travail, vérifiée avant les mises à jour et après
les mesures finales, hors empreintes des preuves. Les tentatives incomplètes
sont conservées et ne sont pas rejouées. Un appel Adam en échec peut avoir
partiellement changé les paramètres : son marqueur reste incertain. Le marqueur
est écrit sur disque à la sortie du try/finally, pas à chaque pas ; un arrêt
brutal laisse l'essai commencé incertain et ne permet aucune relance.

La revue indépendante du résultat flouté doit être committée, puis le protocole,
ses sources et son préflight indépendant doivent être scellés avant le premier
pas. Les tests préparatoires n'exécutent aucune vraie mise à jour d'optimiseur.
Ce diagnostic ne modifie aucun critère ni budget du futur gate Atari.
