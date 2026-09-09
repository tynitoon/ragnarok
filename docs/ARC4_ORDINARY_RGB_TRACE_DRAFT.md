# ARC 4 — images ordinaires et chronologie sans indicateur ALE

Prototype privé séparé des sources scellées, testé uniquement avec un moteur
synthétique. Il ne possède pas encore de transport ni de validation native.
Il conserve la boucle d'action, le tirage sticky, les constructions et les
charges du moteur précédent. Une action ordinaire fournit son image de départ,
ses images réellement produites et son arrivée, sans terminal, RAM, récompense,
nombre de vies, compteur natif ou indicateur de reconstruction.

Le départ est capturé après une éventuelle construction froide, juste avant les
actions demandées. Ses deux buffers RGB respectent le G7 existant : à une
construction, la première image est copiée dans les deux buffers, ce ne sont
pas deux images natives distinctes. Le résultat contient ensuite exactement
une image brute et une image poolée par action native réellement effectuée,
de zéro à quatre, sans compléter les fins tronquées. Ce départ permet de
superviser une transition à partir de la bonne image, même si le moteur a
reconstruit la partie avant d'exécuter la commande.

La chronologie de l'agent compte les images produites par ses actions et livrées
dans ces traces. Les 128/132 images internes de construction, invisibles pendant
le chargement, et les charges synthétiques de restauration restent au registre
de l'expérience ; elles ne deviennent pas des images vues ni un signal de mort.
Cette horloge d'images livrées n'est donc pas le compteur de toutes les images
émulées facturées. Son unité doit être explicitement retenue dans le futur
protocole des objectifs temporels et confrontée aux métriques externes du gate.

Le dernier écran brut observé est comparé à l'écran brut de départ de l'action
suivante. Si ces pixels diffèrent, il y a une discontinuité visuelle entre les
deux morceaux enregistrés : la mémoire locale peut commencer une nouvelle époque.
Ce n'est pas une étiquette de mort. Seul l'écran courant est comparé, pas l'ancien
buffer du pooling : reconstituer un buffer ne crée pas une annotation de reset.
Une reconstruction native invisible dans ces pixels reste indétectable et ne
provoque aucun changement d'époque. Des cibles temporelles erronées restent alors
possibles ; aucune RAM ou vérité terminale ne vient les corriger.

Le collecteur devra posséder ses commandes et lier chaque départ, effet et paire
de branches à la même opération. Sur une discontinuité, le départ est une image
apprise rétrospectivement après la commande, pas une image qui aurait guidé cette
commande avant d'être vue. La décision de jeu ne doit jamais être recalculée
avec ce futur. Le raccord au parser et le choix explicite de cette horloge ne
sont pas encore validés ; aucun gate ou apprentissage n'est lancé.

Une panne après des actions conserve leurs charges et ferme les actions
suivantes. Les refus de budget préalables restent sans dépense. Les sessions
manuelles de snapshots sont refusées par cette API ordinaire ; les paires de
branches continuent d'employer le moteur relu. Les tableaux sont copiés et leur
pooling est vérifié. Le compteur visuel ne valide sa nouvelle mémoire qu'après
la construction complète de son résultat.
