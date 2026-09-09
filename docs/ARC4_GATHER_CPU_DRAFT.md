# Adaptateur natif prospectif du décodage gather

9 septembre 2026. `gather_visible_pool_v70.py` prépare une intégration séparée du
décodage des pastilles par huit pixels. Les méthodes de décodage de chaque image,
groupement, référence, score et branches sont héritées du moteur original ; seule
la fonction choisie pour les scènes MsPacman change avant la première action.
Le cache de groupes de l'essai précédent n'est pas ajouté à cette comparaison.

Le transport garde les deux seules commandes `step` et `report`, draine les
réponses et conserve les rapports après échec. Le garde d'erreur reprend la
précaution du cache : même une allocation de diagnostic échouée bloque la suite.
Les tests sur moteurs synthétiques couvrent trajectoires, reprises froides,
rapports, historiques restaurés, erreur après paiement et deux workers spawn.

Le gain observé sur l'audit offline du décodeur ne prouve pas un gain de la
chaîne complète. L'adaptateur n'est pas encore validé nativement. Il faudra une
revue du résultat offline, un protocole prospectif distinct et sa revue préalable,
puis une vérification native de fidélité avant toute nouvelle mesure complète.
Aucun code gelé, ROM réservé, modèle, cible apprenante ou entraînement ne change.

Le lanceur prospectif `bench_gather_visible_v70.py` et son protocole JSON fixent
les trois cas de raccord19376/19377/19378, puis une seule mesure16×2048 selon
les graines18000..18015, commandes et budgets du natif original. La revue offline
du gather est désormais disponible et exigée, avec ses empreintes exactes. La
référence native antérieure reste conservée. Aucun cache de groupes n'est ajouté.
Les coûts déjà lus sont gardés immédiatement ; un rapport d'échec ou incomplet
reste une borne connue même quand tous les workers ont répondu. Un préflight
indépendant reste nécessaire avant le raccord natif, puis sa revue avant mesure.
