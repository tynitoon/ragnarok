# Inventaire prospectif des essais d'entrée

9 septembre 2026. `entry_trial_plan_v70.py` rend concrète l'arithmétique déjà
décrite dans `ARC4_ENTRY_BUDGET_DRAFT.md`, sans émulation ou entraînement.

Une graine explicite, à inscrire dans le futur protocole, produit une permutation
des dix-huit commandes Atari puis onze commandes distinctes d'une seconde
permutation. Aucun générateur global n'est modifié. Les vingt-neuf essais sont
répartis en deux paires aux treize premiers points courants, puis une paire aux
trois derniers. Il s'agit de points planifiés, sans preuve que seize états
natifs distincts ou suffisamment divers seront rencontrés.

Chaque paire prévoit huit images d'action contre huit images de NOOP et deux
restaurations : jusqu'à 136 charges. Les lots conservent les seuils cumulatifs
du calcul précédent. La simulation des paires complètes retrouve 19 720 charges,
dont 3 944 expérimentales et 58 restaurations. Un autre essai E1/E2/E3/E4/C peut
retarder ou empêcher l'admission : sa dépense n'est jamais soustraite du budget.
Un plafond de jeu supérieur à 20 000 n'autorise pas à dépasser l'entrée de 20 000.

L'inventaire est immuable et ne s'allonge pas si des branches sont tronquées ou
vides. Une commande observée une ou deux fois peut rester entièrement inconnue.
Le module ne lui attribue ni effet nul ni étiquette sémantique. Il ne choisit
pas une graine après résultat et ne prouve pas que cette couverture suffira à
découvrir une table de commandes.

Le contrôle d'admission est en lecture seule sur le budget propre faisant foi.
Il ne débite rien et ne consomme pas un essai : le futur contrôleur doit encore
lier l'inventaire aux opérations uniques du collecteur, gérer les refus et
préserver les échecs payés. L'admission réelle du moteur reste obligatoire.
La sélection des effets, ME, la table, le protocole de collecte et le gate ne
sont pas implémentés ou gelés par cet inventaire.
