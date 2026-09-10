# Clôture de confirmation ARC 3 v62

10 septembre 2026. Préparation de la clôture, sans calcul de score partiel.
Le dernier monde est encore en cours. Ce document applique le protocole gelé ;
il n’ajoute aucun critère, donnée ou essai.

## Avant le score

Attendre les neuf mondes complets et le statut du contrôleur
`confirm_complete_review_required`. Vérifier l’absence des processus de calcul,
les trois fichiers de lignée complets, leur agrégat exact et les checkpoints de
chaque bras. Garder B6, trois mondes par lignée et G' sur les deux premiers mondes
de la lignée0. Les sélections/éventuelles exclusions viennent du nav gate gelé.

Revoir indépendamment les17 sources gelées, les prétrainings M1–M4 des trois
lignées et le probe déjà revu, puis les commandes, reçus et artefacts de
confirmation. Les28 artefacts des lignées0/1 et le M4 de s2_8320 préservés à la
dernière reprise doivent rester exacts. Les bras repris, les coûts des pannes
et leurs traces restent visibles. Aucun remplacement d’un résultat défavorable.

Contrôler le budget avec l’échéance originale09:10:02Z du10 septembre, donc
11:10 Paris, sans refaire partir le compteur après la seconde panne.
Le plafond total de la ligne ARC3 reste42h, pannes facturées comprises.
En cas d’incohérence, préserver les résultats et résoudre la provenance avant
interprétation, sans relancer l’expérience ni modifier le scorer.

## Score et compte rendu

Après revue conforme seulement, exécuter `scripts.score_v61` sur l’agrégat de
production. Vérifier auparavant que `v61_confirm_score.json` n’existe pas : le
scorer gelé utilise une écriture remplaçante. Archiver exactement sa sortie.
La figure corrigée423b770 est déjà revue ; la générer ensuite une fois, puis
inspecter le rendu. Aucun fichier smoke n’entre dans le compte rendu.

Publier le verdict gelé, son motif, Δ_learn et son incertitude, les trois lignées,
les tiers, Δ0 et les écarts finaux. Les résultats et leur interprétation doivent
rester distingués : l’aire des performances après le départ peut conserver
l’effet d’une avance initiale. La règle de formulation tier3 est déjà dans le
scorer ; elle ne transforme pas cette aire en mesure isolée de vitesse.
Présenter les courbes et cette limite ensemble, sans nouveau test après résultat.
Pour v62, Δ0 aux tiers3/4 inclut poids ET store conservé dans le monde.

Le probe M4 a réussi (618/768 contre398/768), mais la confirmation ne mesure pas
l’AUC de la tête de loi sur les cellules exercées des mondes test. Ne pas lui
attribuer une AUC absente ni le mécanisme « table de liaisons transportée »
sur cette seule base. Ce sont des mondes symboliques ; même un verdict positif
ne prouve pas que l’agent joue depuis les pixels ou transfère entre jeux Atari.

Le chapitreARC3_PLAN§12.6 corrige les anciennes commandes et descriptions.
Les règles STOP et les trois catégories du verdict restent inchangées. Toute
suite du projet doit conserver ce résultat et ses limites.
