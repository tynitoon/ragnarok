# Chaîne CPU Pong — diagnostics de développement

Le benchmark traite une vraie requête/réponse avec image RGB par décision,
répète l'action quatre frames, décode chaque frame côté juge et évalue les
instances au dernier état brut. Il crée une paire de branches toutes les
128 décisions, avec admission et coût des deux restaurations. Les cibles C
proviennent de la différence des images, sans RAM.

## Premier essai conservé, à corriger

`cpu_chain_2_workers.json` : 454 décisions/s ; `cpu_chain_16_workers.json` :
1 585 décisions/s, 512 décisions par processus. Initialisation séparée du
chronométrage ; calcul ARC 3 actif. Ces données ne sont pas un verdict de gate.

La revue a identifié deux défauts de portée avant adoption :

1. Le fond de référence était réinitialisé à la frontière d'épisode indiquée
   par l'émulateur. Cette information devait rester dans le juge ; le test de
   décodeur RAM vide ne couvrait pas ce chemin. La référence doit conserver son
   histoire de pixels sans ce reset oracle, ou être explicitement un contrôle
   bénéficiant d'un reset oracle.
2. Les observations ordinaires étaient brutes, mais `branch_pair` renvoyait des
   images max-poolées. Les masques C étaient bien construits depuis les pixels,
   mais sur une représentation différente. Le prochain essai brut devra utiliser
   les images brutes également pour les branches.

Les rapports initiaux, leurs sources et leurs chronométrages restent conservés.
L'intitulé brut du premier JSON ne décrit donc pas entièrement ses branches :
ce défaut est explicité ici, sans modifier rétroactivement le résultat.

La variante max-poolée complète reste une étape ultérieure, avec calibration
temporelle dédiée. La valeur proposée de 2 000 décisions/s pour le gate ne peut
pas encore être jugée depuis ce benchmark sur Pong brut seul.
