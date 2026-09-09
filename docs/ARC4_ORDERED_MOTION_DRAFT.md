# Mouvement : conserver l'ordre des deux écrans propres

Témoin synthétique du 9 septembre 2026, avant tout entraînement Atari. Le parser
relu reçoit actuellement le maximum pixel par pixel des deux derniers écrans.
Cette opération est symétrique : échanger les deux écrans conserve son entrée.
Les archives natives et les diagnostics déjà réalisés restent inchangés.

Le nouveau test construit un carré blanc de2×2pixels sur une image16×16. Dans
un scénario, sa position passe de4 à5 puis continue jusqu'à9 ; dans l'autre, elle
passe de5 à4 puis continue jusqu'à0. Les deux sources max-poolées sont identiques,
mais les cibles poolées après quatre images diffèrent. À hypothèse d'effet égale,
les cinq sorties du parser figé sont exactement égales, vérifié sur CPU seed764.
Les cibles synthétiques servent seulement au témoin ; elles ne sont pas données
au réseau et aucun poids n'est modifié.

La distance L1 entre ces deux cibles, moyennée sur les pixels normalisés, vaut
0,046875. Par l'inégalité triangulaire, une prédiction déterministe commune aux
deux entrées a une perte L1 moyenne au moins égale à0,0234375 sur ces deux cas.
Cela démontre une ambiguïté de représentation possible ; cela ne mesure ni sa
fréquence dans Atari ni une erreur d'apprentissage observée sur ces jeux.

Une entrée qui garde séparément l'écran précédent et l'écran courant distingue
les deux cas sans ajouter d'oracle, de direction calculée, de bouton ou de label.
La correction candidate à préparer est donc une branche de mouvement apprise
recevant ces deux RGB ordonnés, en conservant les images poolées pour les autres
têtes. Les paires existent déjà dans les propres traces ; aucun émulateur ni
nouveau signal externe n'est nécessaire pour exposer cette information.

Cette variante devra être implémentée séparément, testée et soumise à revue avant
de remplacer l'entrée du déplacement dans un futur protocole d'entraînement.
Elle ne supprime pas le problème distinct de découverte des effets des boutons,
ne rend pas les cibles C initiales non nulles par décret et ne prouve pas qu'un
réseau apprendra à suivre la balle. La vision §2.1 décrit encore l'ancienne entrée ;
aucun résultat déjà produit n'est réattribué à la variante candidate.
