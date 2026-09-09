# ARC 4 — capacité de suivi explicite

Le tracker relu ne bornait pas le nombre de pistes conservées pendant les
absences. Plusieurs images comportant chacune de nouveaux objets pouvaient
donc faire grossir la mémoire et la matrice d'affectation avant30frames.

L'adaptateur `bounded_tracker_v70.py` conserve le moteur relu et propose une
limite de512pistes par tracker, absences comprises. Une image de plus de512
instances est refusée avant l'affectation. Si les nouvelles identités et celles
en maintien dépasseraient ce plafond, toute la mise à jour est refusée sans
modifier l'état précédent. Aucun objet n'est évincé silencieusement pour passer
le test. L'expiration normale libère la capacité.

À cette limite, la matrice de coût est au plus512×1024float64, soit4MiB. Les
autres matrices, masques, historiques, copies et identités temporaires s'y
ajoutent ; ce chiffre n'est pas un plafond deRSS. Le candidat peut construire
jusqu'à1024pistes avant le dernier contrôle, puis être abandonné. La configuration
autorise1..4096pistes ; le futur protocole doit fixer sa valeur et mesurer sa
mémoire réelle. Aucun débit ni capacité d'apprentissage n'est établi ici.
