# Raccord prospectif des opérations RGB aux objectifs P/C

9 septembre 2026. Prototype `owned_pixel_objectives_v70.py`, sans collecte native,
optimiseur, politique ni entraînement. Il prend la propriété exclusive d'un
`OwnedRGBCollector` et d'un parser ; aucun appel concurrent à ces objets n'est
admis. La confidentialité Python est un contrat, pas une barrière de sécurité.
Les méthodes publiques portent un verrou non bloquant, couvrant aussi l'inférence
après le retour du collecteur. Un appel concurrent ou réentrant est refusé avant
entrée et laisse l'opération courante se terminer. Le verrou est libéré après
succès ou exception, y compris une erreur d'allocation.

Chaque méthode choisit ses commandes via ses arguments, les fait enregistrer et
exécuter par le collecteur, consomme une seule fois cette opération, puis calcule
les objectifs. Aucun échantillon ou ancienne prédiction fourni par l'appelant
n'est accepté. L'effet passé au parser vient de la copie enregistrée avant
l'action ; le numéro du bouton reste uniquement dans le journal de commande.
Zéro signifie hypothèse inconnue : la correspondance boutons/effets est encore
à apprendre. Le résultat n'atteste pas que l'action a physiquement cet effet.

P utilise le max des deux buffers de départ réels et l'image poolée finale de
l'opération ordinaire. Le départ peut avoir changé après une construction froide
non observée. L'inférence est alors rétrospective : elle ne doit jamais servir à
réviser l'action déjà choisie. Le modèle prédit quatre images d'action livrées.
Les préfixes de zéro à trois images ne supervisent pas P, avec une perte nulle
et un motif explicite ; aucune image manquante n'est fabriquée. Les charges de
construction et de restauration ne deviennent pas du temps visuel observé.

C utilise les deux branches d'une même opération au point courant attesté par
le collecteur. Le parser et les composantes propres sont recalculés sur les
buffers de cet ancrage avec l'hypothèse d'effet de cette opération. C compare les
images **poolées à l'horizon quatre**, y compris pour une paire de huit images ;
les images suivantes ne constituent pas de nouveaux exemples C ici. Un préfixe
commun insuffisant, une absence de mouvement prédit ou une projection entièrement
hors écran restent inconnus. Les cibles supervisent les pixels sources propres.

Toutes les têtes du parser et les pertes rendues doivent être finies. Toute
erreur arrête ce propriétaire, y compris une erreur avant paiement. Si une
inférence échoue après consommation, les enregistrements RGB restent accessibles
en copie par `failure_records`, sans répétition de commande payée. Si `take`
échoue, le collecteur garde son opération en attente. Les rapports de dépenses
sont réservés à l'audit externe. Il n'y a pas encore de conservation durable sur
disque ni de récupération automatique ; la capacité du collecteur et la copie
d'un lot ne bornent pas la mémoire des graphes que l'appelant conserverait.

Ce raccord ne traite pas le tracker, G/I, les catégories ou la mémoire d'identité.
Leur raccord doit définir prospectivement la chronologie des images livrées et
les coupures décidées à partir des pixels. Le champ historique `native_frame`
de `PerceptionStream` ne prouve pas une équivalence avec le compteur natif ALE.
Aucune performance Atari, contrôlabilité apprise ou accélération du transfert
n'est établie par ce prototype. Aucun critère de gate n'est modifié ou gelé ici.
