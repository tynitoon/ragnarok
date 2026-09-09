# ARC 4 — suivi des propres instances, prototype

Le suivi associe les objets prédits dans deux images avec une affectation
hongroise : distance à la position prévue, puis distancecosinus entre embeddings.
La prévision utilise le déplacement du parser, pas une vitesse ou une identitéRAM.
Les paramètres candidats sont8pixels de distance maximale, un poidsd'apparence8,
une mémoirevisuelle4frames et un maintien30frames ; ils restent à geler avant
une collecte. Une candidature hors rayon ne force pas d'association.

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
