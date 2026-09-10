# ARC3 : deuxième interruption, 10 septembre 2026

À 00:26 UTC, les processus du contrôleur et de son enfant n'étaient plus
accessibles. Plusieurs lectures des processus Windows ont confirmé leur
absence. Le dernier démarrage du système était toujours celui du 9 septembre
à 04:39:14 UTC. La cause de cette nouvelle interruption reste inconnue ; aucun
lien causal avec un heartbeat, Windows Update ou le GPU n'est établi.

Les preuves sont conservées dans
`craft_v6_out/v62_confirm_interruption_20260910/`, sans modifier les originaux.
Six mondes complets et le bras M4 complet du monde s2_8320 sont présents. Le
bras Fa n'a pas de checkpoint complet : sa reprise rejoue ses trois objectifs
avec les mêmes graines, en conservant la dépense et le journal déjà produits.
Le script expérimental gelé ne permet pas de reprendre à l'objectif suivant.

Le nouveau contrôleur reprend uniquement la lignée s2 par la commande originale
`--resume`, après validation des 28 artefacts des lignées 0/1 et du M4. Il
préserve les anciens états, les preuves, les sources gelées et le préfixe du
journal. Il exige un reçu indépendant committé correspondant aux empreintes
exactes de ses sources et des preuves. Son état et son lancement sont exclusifs.
Il ne calcule aucun score avant la revue des neuf mondes complets.

Le budget conserve son origine : 9 septembre 09:37:40 UTC, plus
84 742,0902837 secondes. Le contrôleur arrondit l'échéance vers le bas au
10 septembre 09:10:02 UTC (11:10:02 Paris). Toute l'interruption et toute
l'investigation consomment cette fenêtre ; aucun nouveau budget n'est alloué.
L'horloge UTC et une horloge monotone limitent conjointement le temps restant.
Le temps de création du processus est inclus. Une surveillance toutes les
0,5 seconde retire 30 secondes pour arrêter l'enfant et écrire l'état final.
La commande, B6/N9/G2 et les critères scientifiques restent inchangés.

Un probe sans modèle ni émulateur a confirmé qu'un lancement WMI caché
(`Win32_ProcessStartup.ShowWindow=0`) s'exécutait hors Job Object : PID17288,
durée20,00117 secondes après le retour du lanceur. Le probe et ses reçus entrent
dans les empreintes de la reprise. Le lanceur Python et le contrôleur vérifient
chacun cette condition ; le lanceur crée des journaux et un reçu PID exclusifs.
Le contrôleur attend que le reçu `dispatched` nomme exactement son PID et sa
revue avant toute création du processus expérimental. Une publication absente
expire après dix secondes, sans GPU. Une erreur du lanceur après publication
préserve le contrôleur autonome et son délai ; elle ne tue pas le parent en
laissant potentiellement son enfant sans surveillance.
Il n'installe aucun service ni tâche récurrente. Cette précaution répond à la
durée de vie des processus ; elle ne démontre pas la cause de l'interruption.
La mécanique est décrite par Microsoft dans
[Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
et [IsProcessInJob](https://learn.microsoft.com/en-us/windows/win32/api/jobapi/nf-jobapi-isprocessinjob).

Le benchmark CPU rectangle a également perdu son processus. Sa console vide
est préservée et aucun marqueur de tentative ni résultat n'a été trouvé.
Ce constat ne constitue pas, à lui seul, une preuve de coût nul : une revue
séparée des préconditions est nécessaire avant toute reprise de ce benchmark.

Ce document décrit la reprise préparée. Le reçu indépendant, l'état réel et
les processus doivent être vérifiés pour établir si elle a effectivement été
lancée ; la présence de ce document ne signifie pas qu'un calcul tourne.
