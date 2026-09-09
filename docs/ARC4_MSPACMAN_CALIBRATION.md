# Ms. Pac-Man : développement brut

Périmètre initial fixé et revu avant collecte, commit `9714014` : joueur,
fantômes, fruit, grosses et petites pastilles. Cinq catégories séparées,
ROI [0,0,160,171), aucun HUD. La clé ALE est `ms_pacman`. Revue Meitner :
grilles et palettes conformes aux sources MIT épinglées, 504 cas de bits isolés
vérifiés. Trois tests CPU initiaux passent.

Capture 17411 : 8 192 images ordinaires, quatre constructions froides,
8 720 frames facturées. Les 528 frames de préparation sont facturées mais ne
sont pas archivées individuellement. Réservé 17412 toujours fermé.

La baseline nominale V0 échoue sur les cinq catégories, conservée sans changement :

| Catégorie | Présence positive | Boîtes à un pixel près | Conclusion |
|---|---:|---:|---|
| Joueur | 100 % | 36,7188 % | Géométrie nominale incorrecte |
| Fantômes | 11,9629 % | 78,2251 % | Visibilité, contacts et géométrie à vérifier |
| Fruit | null | null | Absent, non calibré |
| Grosses pastilles | 0 % | 49,0853 % | Décor pris pour des pastilles et visibilité à vérifier |
| Petites pastilles | 49,5972 % | 99,2449 % | Comptages différents malgré beaucoup de boîtes proches |

Les petites pastilles représentent plus d'un million d'appariements, mais leur
nombre ne dilue pas les erreurs des acteurs. Aucune correction ni exclusion de
catégorie n'est tirée du réservé. Les prochaines corrections utilisent seulement
ce développement ; une collecte supplémentaire pour les objets absents devra
être définie avant son ouverture. Aucune perception apprise ni compétence de jeu
n'est démontrée par ces mesures d'instruments.

Les diagnostics du 9 septembre identifient des phases de rendu distinctes.
Le candidat séparé `mspacman_render_v70.py` décrit le joueur avec la position
et l'animation de la RAM précédente du même épisode : corps 8×10 à
(RAM10−12,RAM16+1), rogné de 2−RAM55 du côté RAM56 (haut/droite/bas/gauche).
Les quatre premières images, sans RAM précédente archivée, utilisent explicitement
la RAM courante. Vérification sur toutes les images de développement : 8 192
comptages égaux et boîtes à un pixel près, dont 7 913 exactes. Trois tests ciblés
passent. C'est une correction de développement, pas une validation réservée.

Les grosses pastilles suivent les flags117 et la phase30&8 **courants** ; leur
enveloppe nominale ne corrige pas encore les occultations. Les fantômes ont une
liste de rendu distincte de leurs positions logiques ; sa géométrie et le checker
RGB sont en cours de diagnostic indépendant. Les petites pastilles nécessitent
des silhouettes d'acteurs pour prédire les occultations sans consulter les pixels
dans le juge RAM. Le fruit reste absent et non calibré.

Les explorations sont conservées dans les JSON `player_phase_*`,
`player_drawlist_*` et `player_opaque_support_*`. Deux pistes sont rejetées :
RAM34/40/46 est constamment nul (aucune entrée joueur à cet endroit), et un
« nœud rouge » ne figure pas dans les crops propres examinés : le joueur et son
nœud sont jaunes. Le diagnostic de masque jaune|rouge peut être contaminé par
les fantômes rouges ; il ne constitue pas une silhouette opaque validée. Une
boîte correcte ne suffit pas à certifier la forme pixel par pixel.
