# ARC 4 — allocation d'entrée à revoir avant gel

Proposition d'ingénierie du 9 septembre 2026, sans exécution d'un gate ni
augmentation du budget. Elle résout le calcul de capacité ; elle ne prouve
pas qu'un agent identifiera ses commandes avec aussi peu d'essais.

Une paire E1 au point courant coûte au maximum 136 frames facturées :
deux branches de huit frames et deux restaurations de 60. Le balayage
16 points × 18 boutons coûterait 39 168 frames expérimentales, hors les
chargements supplémentaires pour rejoindre des points archivés.

Dans une entrée de 20 000 frames à 20 % d'expériences, il reste au plus
4 000 frames pour **E1, E2, E3, E4 et C réunis**. Même si E1 les reçoit toutes,
seules 29 paires de 136 rentrent. Une trentième coûte 4 080 et ne rentre pas.
Le budget des restaurations ne peut pas être omis pour annoncer une table
complète sur les 16 points.

Une allocation candidate est de deux paires aux treize premiers points
courants et d'une paire aux trois derniers, soit 29 paires sur 16 points.
Une permutation des 18 actions, suivie de onze actions distinctes d'une
seconde permutation, assure une ou deux observations par action. La graine
et l'ordre restent à inscrire dans le protocole final. Les réponses absentes
restent inconnues ; aucun effet nul ne leur est attribué par défaut.

Avec des avances ordinaires de quatre frames et le démarrage facturé de 132,
une simulation pure du `FrameBudget` existant admet ces seize lots avant les
charges cumulées suivantes : 1088, 2448, 3808, 5168, 6528, 7888, 9248, 10608,
11968, 13328, 14688, 16048, 17408, 18224, 18904, 19584. Après le dernier lot :

- 15 776 frames ordinaires facturées, démarrage compris ;
- 464 frames de branche et 58 restaurations, soit 3 944 expérimentales ;
- 19 720 frames facturées au total ; 280 restent avant le plafond d'entrée.

C'est une borne pour des paires complètes, sans autre dépense expérimentale.
Une mort peut tronquer une branche, mais ne justifie ni essai gratuit ni
réallocation choisie après le résultat. Les nouvelles parties restent
facturées et peuvent réduire la quantité d'expérience utile. Atteindre un
ancien point ajoute une restauration à facturer : la proposition évite ce
coût en essayant au point courant, puis en reprenant ce même point.

L'API d'archive persistante, les trajectoires intermédiaires pour mesurer le
délai de contrôle, la sélection des points et le partage avec C restent à
implémenter et revoir. Les jetons `clone` actuels ouvrent une session de
laboratoire ; les conserver ne permet pas de compter la suite comme du jeu
ordinaire. La simple simulation ci-dessus n'autorise pas ce contournement.

Les plafonds du gate 0 (200k pour 0-A, 100k pour 0-B) restent dans les premières
500k frames du jeu : une allocation commune fixe à 20 % suffit pour cette
étape. La décroissance ultérieure vers 5 % demeure à définir avant une étape
qui la traverse ; aucun calendrier implicite n'est ajouté au moteur scellé.
