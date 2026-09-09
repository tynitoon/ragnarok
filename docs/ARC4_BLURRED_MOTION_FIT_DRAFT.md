# Ajustement de la candidate P à plusieurs échelles

Le premier diagnostic s'est terminé avec une erreur native0,025491314 pour le
parser ordonné, supérieure au critère fixé0,01171875. Ce résultat est conservé.
La présente expérience n'en relance aucun bras. Elle évalue séparément la perte
à trois échelles, préparée avant ce premier résultat.

Le même parser ordonné repart des mêmes poids vierges, avec contrôle exact de
leur empreinte et des mesures initiales. Les deux scènes, le budget de256mises
à jour, Adam3e-4 et le plafonnement de norme1 restent identiques. Seul le terme
optimisé remplace la L1 native par la moyenne des L1 de largeur1/3/7. Aucune
pondération supplémentaire, aucun autre objectif, augmentation, scène, graine
ou arrêt anticipé ne sont ajoutés.

Le critère final reste la L1 des images natives, strictement inférieure
à0,01171875. Les pertes filtrées sont rapportées séparément ; une baisse due
au changement d'échelle de l'objectif n'est pas considérée comme un succès.
Les points de mesure restent0/1/8/32/128/256. Le budget CPU est300secondes
coopératives, hors vérifications des preuves, sans GPU ni émulation.

La référence numérique provient du résultat immuable précédent. Aucune mesure
de vitesse des architectures n'est faite. Les deux exemples servent encore
à l'ajustement et à la mesure : pas de généralisation, de compétence Atari,
d'apprentissage accéléré ou de transfert établi. Aucun poids appris ici n'est
conservé pour être réutilisé dans un jeu.

La revue de la perte candidate, celle du résultat initial et un préflight
indépendant de ce protocole sont requis avant toute mise à jour. Un seul essai
est admis, ses échecs et son résultat négatif éventuel sont conservés. Le pas
d'optimiseur en cours est marqué en mémoire avant l'appel et persisté en finally
sur exception ; un arrêt dur laisse la tentative incertaine et interdit un
nouvel essai. Aucun critère ni résultat du gate Atari n'est changé.

Un test initial exigeait à tort une égalité bit à bit entre la réduction globale
du terme natif de la candidate et la moyenne des deux erreurs par scène de la
métrique conservée. Leur différence observée était3,73e-9 en float32. Ce seul
contrôle croisé utilise désormais une tolérance absolue1e-8 ; les valeurs
initiales de la métrique native restent exigées strictement identiques à la
référence, et le critère final reste inchangé. Aucun optimiseur réel n'a été
exécuté pendant ces tests.
