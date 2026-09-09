# Validation prospective du transport RGB entre interpréteurs

Ce contrôle technique utilise le Python système du modèle comme parent et le
Python ARC4 explicitement identifié comme enfant. Aucun modèle, Torch ou runtime
ALE n'est importé dans le parent ; la présence de Torch est relevée dans les
métadonnées de distribution. L'enfant connaît uniquement les six jeux de
développement. Son accueil fournit le runtime attendu avant toute initialisation.

Le protocole JSON fixe neuf cas : les six jeux avec la graine19200, puis les trois
préfixes réels de Pong déjà archivés. Les commandes, limites, probabilités sticky
et points d'expérience sont exactement ceux de la validation RGB ordinaire
précédente. C'est une répétition déclarée pour comparer un transport différent,
pas une nouvelle population indépendante ni une tentative d'améliorer un score.
L'ancienne exécution ne sera pas relancée ou modifiée.

Chaque cas dispose d'un seul nouvel émulateur. Les1266pas sont vérifiés par leur
empreinte ordonnée complète source/raw/pooled/endpoint. Les huit positions déjà
archivées par cas sont comparées pixel par pixel, ainsi que les images complètes
des six expériences, leurs comptes, les retours au point de départ et la suite
ordinaire. Les identifiants de session sont normalisés seulement pour comparer
les métadonnées à la référence ; ils ne sont jamais des entrées d'un modèle.

Le succès exige les neuf empreintes, métadonnées et comptes exacts, soit7454
charges attendues. La borne matérielle est neuf constructions avec2000charges
maximum chacune. Le délai de travail est300secondes, vérifié avant chaque
commande et à la fin, avec30secondes par lot de transport ; c'est une limite
coopérative qui n'inclut pas les vérifications des empreintes de sources. Aucun
débit n'est mesuré. Ce contrôle ne concurrence aucun autre essai CPU natif.

Le marqueur d'essai, les résultats et les coûts connus sont conservés même en
échec ; aucune substitution de graine ni nouvel essai n'est permis. Chaque pool
est fermé dans un finally, puis son processus et ses threads sont vérifiés
arrêtés. Une construction ou réponse incertaine laisse un coût explicitement
incomplet. La fermeture après arrêt brutal du système n'est pas garantie ici.

Les sources, références et runtimes doivent être committés et revus avant la
première initialisation. Le résultat sera relu avant intégration du modèle.
Aucun entraînement, optimisation, inférence, GPU ou résultat de gate n'est prévu.
La vérification séquentielle d'un seul worker par cas ne mesure pas un pool
natif de16workers. Les casT0 restent couverts uniquement synthétiquement.
