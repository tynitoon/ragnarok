# Ajustement prospectif du réseau avec priorités prédites

Le défaut de recouvrement du reconstructeur uniforme est établi sur un jouet
opaque ; sa correction numérique et la tête de priorité sont des candidats
séparés. Cet essai unique examinera si le nouveau réseau peut ajuster les deux
scènes déjà déclarées. Il ne réexécute aucun des essais précédents et ne teste
pas les cibles de premier plan ou de contrôle, encore ouvertes.

Avant le premier pas, le résultat à champs libres, l'opérateur corrigé et la
tête doivent avoir leurs revues indépendantes committées. Les sources complètes,
ce protocole et son préflight indépendant doivent aussi être scellés. L'entrée
reste la paire RGB passée, sous effet UNKNOWN ; seule l'image future supervise
la perte. Elle ne fournit ni priorité, déplacement, masque ou identité au réseau.

Une seule graine, 764 ; 256 mises à jour Adam, lot de deux scènes, lr3e-4,
β=(0,9 ; 0,999), ε1e-8, weight decay nul, norme des gradients bornée à1. Les
mesures restent aux pas0/1/8/32/128/256. L'objectif est uniquement la moyenne
des L1 aux largeurs1/3/7, avec le nouveau reconstructeur. Aucun objectif m/C/G/I
ni donnée supplémentaire n'est ajouté. Les têtes visuelles inutilisées peuvent
changer indirectement via les caractéristiques partagées ; elles ne sont pas
directement entraînées par une cible de premier plan ou de contrôle.

Les poids vierges du sous-modèle visuel doivent être exactement ceux du réseau
ordonné original, et les log-priorités doivent commencer exactement à zéro.
La nouvelle accumulation pondérée utilise float64 ; la tolérance absolue des
mesures natives initiales face aux valeurs originales est donc déclarée à2e-7,
avant tout ajustement. Elle concerne seulement ce contrôle de départ. Le critère
final reste **L1 RGB native strictement inférieure à0,01171875**, sans tolérance.
Le filtrage et la nouvelle priorité ne changent pas ce seuil.

Les champs, priorités, couvertures et reconstructions finaux seront conservés
pour l'analyse. Un succès ne démontrerait ni mouvement physique correct, ni
priorités ou visibilité calibrées. Une amélioration ne pourrait pas être
attribuée à la seule priorité : architecture et calcul du reconstructeur ont
changé. L'ajustement et la mesure portent sur les mêmes deux scènes, donc aucun
test de généralisation, jeu Atari, transfert ou apprentissage accéléré n'est fait.

CPU un thread, sans GPU ni émulation. Borne de300s de travail hors empreintes,
contrôlée avant les pas et après les mesures finales. Aucun nouveau tirage,
allongement ou retry. Les échecs et les appels d'optimiseur incertains restent
conservés. Le marqueur de pas est maintenu en mémoire et persisté en finally ;
un arrêt dur laisse une tentative commencée incertaine, sans autoriser de reprise.
Les tests de protocole ne réalisent aucune vraie mise à jour d'optimiseur.
