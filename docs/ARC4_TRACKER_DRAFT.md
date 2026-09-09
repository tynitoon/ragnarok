# ARC 4 — suivi des propres instances, prototype

Le suivi associe les objets prédits dans deux images avec une affectation
hongroise : distance à la position prévue +8fois la distancecosinus des embeddings,
une somme pondérée plutôt qu'une priorité lexicographique.
La prévision utilise le déplacement du parser, pas une vitesse ou une identitéRAM.
Les paramètres candidats sont8pixels de distance maximale, un poidsd'apparence8,
une mémoirevisuelle4frames et un maintien30frames ; ils restent à geler avant
une collecte. Une candidature hors rayon ne force pas d'association.
Si la fenêtre de mémoire est vide, le dernier embedding observé sert de repli.
Les centres sont conservés en pixels depuis les boîtes exactes, sans aller-retour
parfloat32. La tolérance numérique du déplacement est1e-12pixel plus l'epsilon
float32 fois la norme du déplacement prédit écoulé. Elle représente l'arrondi
du descripteur, pas une extension empirique du rayon. Les coûts d'affectation
sont normalisés par une même échelle positive pour garder les valeurs finies.

L'appelant fournit une horloge croissante et l'horizon du déplacement prédit,
dans la même unité. Le tracker ne demande aucune horloge à l'émulateur. Le
collecteur devra définir comment il compte les avances ordinaires et les traces
tronquées ; fournir systématiquement4 pour une vraie durée1 ne serait pas une
vitesse native correctement mesurée. Rien ne convertit implicitement une
décision en frame. Un changement de résolution demande un nouveau tracker.

Les objets réellement observés et ceux seulement prédits pendant une absence
sont rendus dans deux listes distinctes. Le maintien d'un objet absent ne peut
donc pas devenir une détection visible dans le scorer. Après30frames d'absence,
son identitéexpire ; un nouvel objet reçoit une nouvelle identité. Les
réapparitions avant expiration sont signalées.

Une copie indépendante de la mémoire du tracker permet une branche de
laboratoire sans modifier le suivi du jeu ordinaire. Les descripteurs retenus
sont copiés. Les tests couvrent déplacement rapide, croisement, permutation de
la liste d'objets, absence, retour, expiration, branche et arguments invalides.
Ils ne mesurent aucune identité persistante sur Atari, ni compétenceME.
Chaque mise à jour est préparée dans une copie, puis validée d'un bloc. Une
erreur d'allocation ne laisse pas d'objets avancés derrière une ancienne horloge.
Ce coût de copie reste à mesurer avant un benchmark du futur agent.
