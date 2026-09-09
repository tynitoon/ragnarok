# ARC 4 — traces d'expériences, prototype CPU du 9 septembre 2026

Le moteur actuel rend seulement l'image finale de chaque branche. Cela ne
suffit pas pour vérifier le délai d'un effet : déplacement direct du joueur,
puis conséquence sur une balle, par exemple. La distinction reste une
hypothèse à mesurer ; obtenir des traces ne la démontre pas.

`ragnarok/environments/rgb_branch_trace_v70.py` ajoute un adaptateur privé du
moteur froid. La boucle native, le tirage sticky, l'arrêt terminal, les
restaurations et le registre de charges restent ceux du moteur inchangé.
Une file temporaire conserve chaque image déjà facturée, ainsi que le maximum
des deux dernières images. Les deux branches rendent chacune leurs traces RGB
et leur image finale. Aucun état natif, RAM, récompense ou nombre de vies ne
figure dans cette sortie. Une branche tronquée a une trace plus courte ; elle
n'est pas complétée par des images fictives.

Une expérience E1 de huit frames par branche pourrait fournir aussi le préfixe
de quatre frames utile à C, sans deuxième expérience ni restauration. Ce
partage, le traitement des préfixes de longueurs différentes et l'unité du
délai de contrôle restent à fixer avant le protocole de collecte. Le prototype
n'autorise aucun apprentissage sur les captures de calibration.

Cinq tests synthétiques vérifient les images brutes et poolées à chaque frame,
l'identité des sorties avec le moteur précédent, la restauration des RNG
natif/sticky et de l'historique RGB, les traces de longueur zéro ou un, les
charges 120/122/128 et le refus avant dépense si le budget est insuffisant.
Une panne de l'enregistrement après une frame conserve sa charge et celle de
la restauration, puis interdit les actions ultérieures. Les tableaux retournés
ne partagent pas la mémoire du moteur restauré.

Il manque encore la revue indépendante, le contrôle avec les vrais émulateurs,
le transport fermé des traces et l'archive persistante. Aucun nouveau jeu ou
nouvelle graine n'a été ouvert pour ce prototype, qui reste séparé des
115 sources du benchmark CPU de référence en préparation.

`rgb_trace_difference_v70.py` compare ensuite uniquement le préfixe réellement
observé dans les deux branches, avec un choix explicite brut/poolé. Le premier
changement est numéroté en frames natives depuis l'ancrage ; zéro veut dire
« aucun changement observé dans ce préfixe », pas absence de contrôlabilité.
Les queues tronquées sont comptées séparément. Sept tests synthétiques couvrent
ces cas et l'effet de persistance du pooling. Aucune décision ME ni identité
d'objet n'est dérivée de ces seuls pixels. La provenance commune des deux
branches doit encore être imposée par le transport et le protocole de collecte.
