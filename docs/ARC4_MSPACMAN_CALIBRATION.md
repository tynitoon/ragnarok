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
