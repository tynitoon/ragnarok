# Raccord temporel prospectif à la collecte possédée

9 septembre 2026. `owned_perception_session_v70.py` assemble les opérations RGB,
P/C et les mémoires `PerceptionStream` de chaque worker. CPU synthétique seulement,
aucune collecte native, politique, mise à jour des poids ou mesure Atari.

Le constructeur prend la propriété exclusive du collecteur et du parser. Il
amorce chaque mémoire sur les deux buffers RGB courants, puis abandonne le graphe
initial ; cette initialisation n'est pas comptée comme une nouvelle action.
Une méthode `step` exécute et consomme une opération via `OwnedPixelObjectives`.
Ses objectifs P/C conservent leur hypothèse d'effet enregistrée avant paiement.
Les observations temporelles en fin d'action utilisent l'hypothèse **inconnue**
(douze zéros) pour la prochaine action. Le déplacement du tracker est donc une
prédiction propre incertaine sous cette hypothèse, pas un mouvement mesuré ni
l'effet réel du bouton précédent. La correspondance boutons/effets reste absente.

## Convention temporelle explicite, à décider avant le gate

Le champ historique `native_frame` du résultat `PerceptionStream` reçoit ici le
compteur local des **images d'action ordinaires livrées**. Ce n'est pas le
compteur ALE, ni le budget facturé : les images non rendues de construction
froide, les restaurations et les branches ne vieillissent pas cette mémoire.
I compare uniquement des observations exactement séparées de huit unités de
cette horloge ; aucune substitution « deux appels = huit images » n'est faite.
Cette convention prospective n'est pas une validation du lag natif du gate.

Une différence entre le dernier écran brut et le départ réel de l'opération
coupe la mémoire, avec une nouvelle époque locale. La nouvelle mémoire est
amorcée au départ réel, puis reçoit l'arrivée si au moins une image d'action
a été livrée. Aucun terminal, compteur de vies ou événement RAM ne décide cela.
Un reset qui laisse les mêmes pixels est indétectable : des identités peuvent
alors être maintenues à tort. Une rupture visible n'atteste pas une mort.

Un préfixe vide sans rupture visuelle rend `observations[worker] = None` : ni
avancement, ni nouveau G/I, ni répétition du graphe précédent. Avec rupture mais
zéro action, seule l'image d'ancrage de la nouvelle mémoire est rendue, à zéro,
sans cible I ancienne. P reste inconnu dans les deux cas. Si seul le buffer
poolé ancien change mais que l'écran brut courant reste égal, cela ne déclenche
pas de reset ; sans nouvelle action, la mémoire précédente reste conservée.
L'autorité des pixels courants pour une future décision reste `current_anchor`,
pas une ancienne sortie de perception. Les branches ne touchent jamais la
mémoire ordinaire, son horloge, ses identifiants ou son historique I.

## Atomicité et limites

Un verrou non bloquant couvre chaque méthode publique jusqu'à la fin du calcul.
Une enveloppe superficielle de chaque `PerceptionStream` partage temporairement
ses anciennes entrées : la version relue de `observe` ne les modifie pas et
prépare un nouveau tracker/historique avant commit. Les poids ne sont pas copiés.
Les mémoires de tous les workers sont remplacées ensemble, une fois le lot
entièrement construit. Une erreur après paiement arrête la session, garde ses
anciennes mémoires et conserve les enregistrements payés accessibles en copie.
Les graphes du lot en attente peuvent rester en mémoire après cet arrêt : ce
n'est pas un plafond RSS, ni une conservation durable sur disque. Les rapports
restent réservés au diagnostic externe. Les accès directs aux objets possédés
ou aux poids pendant un appel sont exclus par le contrat, pas par une sécurité
Python. Toute évolution de `PerceptionStream.observe` exige de revoir l'atomicité
de cet adaptateur.

Pas de catégories comportementales, ME, lecteur de score, mort apprise, pulsions,
politique, optimiseur, ledger durable ou export de compétences dans ce module.
