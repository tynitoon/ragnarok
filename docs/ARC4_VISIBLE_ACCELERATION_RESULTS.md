# ARC 4 — équivalence des adaptateurs du juge, 9 septembre 2026

Audit offline terminé sous le protocole committé `860319d`, avant son exécution.
Les 40965 paires de sources des collections DEV, réservée et brute fraîche sont
identiques au décodeur précédent : 3613727 labels ordonnés, RAM inchangée, mêmes
trois empreintes que l'audit antérieur. Les 24576 regroupements ont également
la même géométrie, le même ordre et les mêmes appartenances aux objets sources.

| Calcul, somme des trois collections | Ancien | Candidat |
| --- | ---: | ---: |
| Décodage des objets | 31,139 s | 32,634 s |
| Regroupement des supports | 47,457 s | 32,243 s |

La vectorisation par image intégrale est plus lente au total, malgré un gain
sur la collection brute fraîche. Ce résultat est conservé ; cette version ne
remplace pas le décodeur dans une nouvelle chaîne. Le cache des regroupements
est une piste à retenir, sous réserve de revue indépendante et d'une mesure
complète prospective. Aucun seuil scientifique n'est modifié par ce choix
d'ingénierie sur des données de développement déjà ouvertes.

Ces temps sont des appels séquentiels à des fonctions sur des archives, sans
profileur, sous charge ARC 3. Ils n'incluent ni émulation, ni IPC, ni restaurations,
ni apprentissage. L'ancien débit MsPacman de 580,793 décisions/s reste le dernier
débit natif mesuré. Aucune extrapolation ne le remplace, aucun gate n'est lancé.

Résultat : `craft_v6_out/v70_visible_cpu/vector_cached_equivalence_v0.json`,
SHA256 `9f13705297c1517a1ca9156ef2636938c29bc7a3fa022855921b432cff05d8ce`.
Le fichier de tentative est conservé à côté ; la revue indépendante reste à faire.

Limite supplémentaire : cet audit n'a pas inscrit son runtime dans son résultat.
Le runtime d'une contre-vérification ultérieure doit donc être présenté séparément,
sans le faire passer pour celui mesuré. L'équivalence exhaustive du cache concerne
ici MsPacman ; les vérifications Asteroids restent synthétiques.
