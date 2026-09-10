# Vérification algébrique séparée des coefficients appris

10 septembre 2026. Déclaration après la clôture du fit affine3ef0cf9, avant
lecture des coefficients par ce diagnostic. Les résultats observés sur le lot
sont connus : Newton IoU1, Adam IoU0,56577, tous deux AP1 et zéro faux positif.
Ils ne servent pas à changer les seuils du fit. Sa revue indépendante doit
être conforme avant cette nouvelle vérification, qui ne relance aucun calcul
d’apprentissage et ne modifie aucune sortie précédente.

## Question et domaine exact

L’association apprise pourrait-elle dépendre des douze positions ou du bruit
vert particulier du lot ? Le modèle affine partagé n’a ni coordonnées ni
identifiants. On peut examiner exactement ses décisions sur toute la famille
stationnaire déclarée, sans tirer de nouvelles images.

Dans cette famille, les deux passés avant intervention sont égaux ; les états
après intervention sont hors de ce domaine. Les valeurs RGB normalisées
sont comprises entre0 et1. Sur les pixels dont l’intervention change le bleu,
R=1, B=0, G est quelconque. Sur les autres, R=0, G et B sont quelconques.
Ces propriétés viennent du générateur certifié et du calcul des différences
de branches ; elles sont réservées à ce juge. L’apprenant ne reçoit ni cette
règle, ni les coefficients du certificat construit auparavant.

Pour les sept coefficients appris dans l’ordre R0,G0,B0,R1,G1,B1,b, définir
aR=w0+w3, aG=w1+w4 et aB=w2+w5. Le logit exact vaut
`b + aR*R + aG*G + aB*B`. Ses extrema sur les deux classes sont donc :

- minimum sur les pixels positifs : `b + aR + min(0,aG)` ;
- maximum sur les pixels négatifs : `b + max(0,aG) + max(0,aB)`.

Les extrémités G/B=0 ou1 appartiennent au domaine. Les calculer en convertissant
chacun des sept float64 sauvegardés en rationnel exact AVANT toute addition,
sans multiplication
matricielle sur des observations. Le certificat strict est positif si le
premier minimum est>0 ET le second maximum est<0. Le seuil logit0 correspond
mathématiquement à la probabilité0,5 déjà fixée ; aucune recherche de seuil.

Une égalité donne « non certifié » par ce test strict, même si la convention
>=0,5 pourrait classer correctement certaines égalités. Les bornes rationnelles
portent sur la fonction affine exacte définie par les coefficients stockés :
elles ne prétendent pas vérifier tous les arrondis d’un noyau BLAS ou sigmoid.
La revue du fit vérifie séparément les sorties effectivement calculées sur le lot.
Seules les sommes des coefficients temporels sont identifiables dans ces passés
identiques ; ce certificat n'établit aucune dépendance temporelle apprise.

## Exécution et interprétation

Examiner une fois chacun des quatre bras appris et l’initial commun, dans
l’ordre initial/Newton visuel/Newton constant/Adam visuel/Adam constant. Publier
les cinq bornes et verdicts, avec Newton visuel primaire. Aucun bras ne remplace
ce verdict après résultat. Les constantes restent des témoins ; aucun certificat
de référence fourni au learner n’est ouvert ni construit ici.

Une exécution de juge CPU1,60s dont10s de réserve,1MB de sorties ; aucune image
générée ou décodée, zéro émulation, produit de modèle, gradient ou mise à jour.
Vérifier les empreintes du fit et de sa revue, les deux NPZ de coefficients et
leurs signatures ; les fichiers sources nouveaux sont gelés et revus avant
exécution. Les deux archives initial/final peuvent être validées une seule fois
chacune ; chacune contient plusieurs tableaux, dont le nombre validé est publié.
Conserver les erreurs et résultats partiels ; aucune réécriture ou retry.
Temps des vérifications, décodage, arithmétique, provenance finale et payload
fsync/hash compris ; seul le reçu final et la console sont hors mesure.

Un certificat établirait une règle apprise couvrant cette famille de couleurs,
positions et bruits, dans le modèle mathématique annoncé. La règle synthétique
reste très simple et connue du juge. Cela ne prouve ni perception des objets
Atari, ni robustesse hors de cette famille, ni maîtrise d’un jeu, ni transfert
ou accélération. L’ancienne validation conditionnelle775 demeure fermée.
