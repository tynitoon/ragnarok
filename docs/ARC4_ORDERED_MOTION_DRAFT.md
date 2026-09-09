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

Variante maintenant implémentée séparément : `OrderedMotionParser` garde la
pyramide native/demi-résolution et les têtes f/m/c/s sur l'image poolée. Une branche
apprise6→32canaux reçoit les deux écrans RGB dans leur ordre ; le déplacement
combine ses32canaux avec les32canaux poolés et les32canaux d'hypothèse d'effet.
Ce sont uniquement des convolutions apprises, sans différence ou flux fourni
comme cible. L'appel impose deux écrans et refuse l'ancienne signature, afin
qu'un ancien lanceur ne perde pas silencieusement cette information.

Les tests synthétiques vérifient que l'ordre peut changer d sans changer f/m/c/s,
que l'effet ne conditionne toujours que d, que P transmet un gradient à la
nouvelle branche et que les poids restent inchangés sans optimiseur. Une sortie
différente du réseau vierge ne constitue pas une direction correctement apprise.
L'initialisation du dernier déplacement reste petite ; cette variante ne force
pas l'existence de cibles C dès le départ. Aucune vraie archive n'est utilisée
pour ces tests et aucune inférence native de cette variante n'est encore lancée.
