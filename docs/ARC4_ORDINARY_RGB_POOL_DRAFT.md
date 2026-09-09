# ARC 4 — transport des traces ordinaires

Le nouveau transport est séparé des sources scellées. Il accepte seulement
les observations RGB, la paire de buffers RGB, les actions ordinaires, les
paires de branches existantes, les comptes externes, le rapport et la fermeture.
Il ne rend ni RAM, ni récompense, ni vies, ni horloge native, ni drapeau terminal,
ni snapshot. Une action ordinaire rend la trace avec son départ réellement
enregistré, ses images brutes/poolées et son arrivée. Le parent vérifie le pooling
et le format avant de remettre ces images au futur collecteur.

Une erreur d'action ou une réponse mal formée ferme les actions du parent,
y compris vers les autres workers, sans renvoyer une commande ambiguë. Les
rapports et la fermeture restent disponibles quand le transport permet encore
de les demander. Une perte de processus ou de canal peut rendre une partie des
charges inconnue ; le futur lanceur doit conserver les tentatives et les
rapports déjà reçus, jamais inventer une absence de dépense.

Quatre tests vérifient la liste fermée des commandes, les charges après panne,
le refus d'une mauvaise réponse et un vrai échange entre deux processus Windows
avec moteurs entièrement synthétiques. Ce dernier test vérifie les buffers
de départ, les traces, les restaurations et la fermeture des processus. Il ne
constitue pas une validation avec les émulateurs réels.

Le propriétaire du futur collecteur devra encore associer les identifiants de
requêtes et les effets à chaque opération, et empêcher de consommer deux fois
une réponse comme s'il s'agissait de deux expériences. Ces garanties ne sont
pas apportées par la seule sérialisation des tableaux. Aucune collecte native,
aucun apprentissage ni aucun gate n'est lancé par ce prototype.
