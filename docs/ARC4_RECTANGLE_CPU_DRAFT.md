# Adaptateur CPU distinct pour le groupement par rectangles

Candidate du 10 septembre 2026, sans lancement natif. Le groupement hors ligne
est terminé, sa revue numérique reste un préalable. Le gain de ce composant
ne prédit pas celui du système complet.

La candidate MsPacman réutilise le chemin step, stop et report déjà revu de
`CachedVisibleEngine`. Après sa construction et avant toute scène produite par
une commande, elle remplace son objet de groupement par une instance sans état,
qui appelle `rectangle_visible_groups`. Le cache d'entrée/résultat n'est donc
pas ajouté à cette comparaison. Le décodeur original reste sélectionné : la
variante gather n'est pas combinée non plus.

Les pixels, référence, historique brut du juge, scores, branches, restaurations,
empreintes et coûts continuent à suivre le même code. Le verrou après erreur
reste actif même si l'allocation du diagnostic d'erreur échoue. Le transport
garde uniquement step et report, draine les réponses et ferme les workers.

Les tests utilisent des moteurs synthétiques : trajectoire ordinaire, reset
froid, branches, égalité des rapports, erreur après paiement et deux processus
spawn. Les jeux extérieurs à MsPacman sont refusés avant construction native.
Asteroids a participé à la comparaison du composant hors ligne, mais n'est pas
inclus implicitement dans cet adaptateur ni dans une future mesure native.

Aucune source gelée ni aucun ancien appelant n'est modifié. Une revue de
l'adaptateur, un protocole de raccord fidèle et une revue préalable seront
nécessaires avant toute émulation ; la mesure complète restera une étape distincte.
