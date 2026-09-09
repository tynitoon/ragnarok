# ARC 4 — transport privé de l'archive

Le prototype `rgb_archive_pool_v70.py` ajoute au transport les jetons d'archive
et les paires de traces. Les commandes sont une liste fermée : images, actions,
branches, sauvegarde, suppression, paire sur archive et rapport d'instrument.
Aucun état natif, import d'état, RAM, récompense ou nombre de vies ne traverse
cette interface. Les anciens appels clone/restore/release restent refusés.

Les plafonds de20000entrées et2milliards d'octets sérialisés s'appliquent au
**pool entier** : chaque worker reçoit la division entière de chaque plafond
par le nombre de workers. Les restes demeurent inutilisés. Ce partage fixe
n'autorise pas un worker à emprunter la mémoire d'un autre. Comme pour le moteur,
il s'agit du payload conservé, pas de la mémoire totale du processus.

Après tout échec de sauvegarde ou de paire, le parent interdit les actions
suivantes ; les rapports et suppressions restent disponibles. Ce choix est
plus strict que le refus récupérable de capacité du moteur privé : le futur
collecteur devra vérifier sa capacité et choisir ses suppressions avant l'appel.
Une suppression ne rouvre jamais une trajectoire arrêtée.

Les essais utilisent des workers Windows réels mais des émulateurs synthétiques.
Une validation native prospective, la politique de cellules et le retour durable
E4 restent à réaliser. Aucun lancement natif ni entraînement n'est autorisé ici.
