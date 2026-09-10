# Voie RGB directe : la recette CNN reste insuffisante

10 septembre 2026. L’essai unique a terminé ses512mises à jour en
756,819768secondes sur un CPU, sans erreur technique. Les trois clauses
scientifiques échouent. Les résultats sont scellés2b2de7e ; la revue
indépendante f3f98ce confirme les tableaux, les compteurs et leur provenance.

## Comparaison déclarée

Le CNN de10177paramètres reçoit une voie RGB1×1 affine de sept paramètres,
initialement nuls, additionnée à ses logits. Le candidat de10184paramètres
garde le même initial CNN et la même recette Adam3e-4/clip1/float32,
256mises à jour visuelles et256sur entrée constante. Aucun poids du certificat
ni de la calibration affine n’a été importé. Les signatures des poids CNN,
logits, probabilités et mesures initiaux sont identiques à l’ancienne archive
avant toute optimisation.

Les douze exemples artificiels sont ceux du contrôle stationnaire déjà
construit. Aucun écran, pas synthétique, image Atari, jeu ou passage GPU
supplémentaire n’a été produit. Le protocole est
docs/ARC4_STATIONARY_DIRECT_PATH_DRAFT.md, sources75278b2,
préflight indépendant948148a.

| Mesure de la sortie complète | Candidat direct | Ancien CNN | Critère candidat |
|---|---:|---:|---:|
| IoU moyenne | 0,007362 | 0,004930 | ≥0,25 : FAIL |
| Avance sur le meilleur témoin | 0,004841 | 0,002409 | ≥0,10 : FAIL |
| Faux positifs, premier cas vide | 43/33600 =0,127976% | 78/33600 | ≤0,1% : FAIL |
| Faux positifs, second cas vide | 71/33600 =0,211310% | 158/33600 | ≤0,1% : FAIL |
| AP moyenne | 0,207505 | 0,203889 | Descriptif |
| BCE équilibrée finale | 0,516230 | 0,558961 | Descriptif |

L’ajout améliore modestement ces mesures, mais ne rend pas cette recette
suffisante sur le lot. Le témoin constant apprend une sortie négative
(IoU0, aucun faux positif vide, BCE0,679193). Le meilleur témoin selon l’IoU
est l’initial,0,00252083, qui prédit partout positif. Les temps historiques
ont été obtenus avec des charges différentes ; aucune accélération de
l’exécution ne se déduit de leur comparaison.

## Tous les cas, au seuil fixé de0,5

| Cas | Pixels cible | Vrais positifs, somme | Faux positifs, somme | IoU somme | Faux positifs, CNN seul | Faux positifs, direct seul |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 315 | 0,003165 | 337 | 954 |
| 2 | 4 | 1 | 189 | 0,005181 | 217 | 990 |
| 3 | 16 | 3 | 549 | 0,005310 | 623 | 967 |
| 4 | 64 | 4 | 603 | 0,005997 | 730 | 948 |
| 5 | 128 | 31 | 2780 | 0,010660 | 3612 | 912 |
| 6 | 256 | 156 | 10448 | 0,014574 | 12258 | 961 |
| 7, vide | 0 | 0 | 43 | — | 65 | 934 |
| 8, vide | 0 | 0 | 71 | — | 119 | 924 |
| 9 | 336 | 292 | 21351 | 0,013464 | 23676 | 956 |
| 10 | 32 | 4 | 627 | 0,006070 | 735 | 946 |
| 11 | 8 | 3 | 459 | 0,006424 | 487 | 919 |
| 12 | 2 | 1 | 359 | 0,002770 | 394 | 994 |

Les moyennes sont calculées par paire, puis entre les cinq paires non vides.
Les deux cas vides restent chacun une condition obligatoire.

## Ce qu’apportent les sorties séparées

| Contribution du bras visuel final | IoU moyenne | AP moyenne | BCE | Faux positifs vides |
|---|---:|---:|---:|---:|
| Somme, seule sortie jugée | 0,007362 | 0,207505 | 0,516230 | 43 /71 |
| CNN seul | 0,005249 | 0,203932 | 0,558523 | 65 /119 |
| Voie directe seule | 0,072144 | 1,000000 | 0,642814 | 934 /924 |

La voie directe retrouve les847pixels positifs, avec une AP de1 sur chaque
cas non vide, tout en prédisant beaucoup trop de négatifs comme positifs.
Une bonne AP exprime un bon classement des pixels ; elle ne garantit pas
la décision au seuil fixé. Aucun seuil n’est recherché après observation.
La somme retrouve496/847positifs et échoue aux conditions de faux positifs.

Ces contributions ont été enregistrées pendant les trois évaluations déjà
comptées, sans nouvelle inférence. Leur retrait décrit la fonction finale.
Il n’isole pas leur rôle causal dans l’apprentissage : gradients et clipping
sont partagés, les fonctions coévoluent et les deux biais peuvent se compenser.
Les performances de la voie directe retirée ne sont donc pas celles d’un
modèle affine qui aurait été entraîné seul.

![Les douze cas et les contributions conservées](F:/dev/ragnarok/craft_v6_out/v70_stationary_direct_fit/fit_v0.png)

La figure a été produite à partir des seuls JSON et du journal, sans ouvrir
de tableaux ni effectuer d’inférence supplémentaire ; son rendu a été inspecté.

## Conservation et portée

Le lancement WMI caché et séparé du processus de l’outil est documenté dans
craft_v6_out/v70_stationary_direct_fit/launch_v0/. Le PID12332 s’est arrêté.
Les compteurs exacts donnent512forwards d’entraînement et trois d’évaluation,
512mises à jour et aucune interruption. Les14artefacts binaires, paramètres
initiaux et finaux, logits, probabilités, contributions, journal et reçus
sont préservés. RésultatSHA :
cea34fdcdc5f1de67e6edc595184a6b961770a018700874fc2a3461f5d6b37b5.

Le réviseur indépendant a contrôlé119sources,37preuves,14artefacts binaires
(12301925octets),2055événements et120lignes de métriques avec9BCE en7,40s.
Il a relu les19tableaux et les deux checkpoints sauvegardés, sans construire
ni appeler de réseau. L’initial identique, la voie nulle et les sommes exactes
sont confirmés. ReçuSHA :
723596fefb4a094c38053d9cd40b019abd57c5f7f69b202b86f51a21e8bef978.

Cette recette se clôt avec son résultat négatif. Les validations conditionnelles
antérieures restent non exécutées. Le succès affine indépendant sur la famille
artificielle demeure un résultat distinct. Ni ce contrôle ni son amélioration
numérique ne démontrent une maîtrise de jeu, un transfert ou un apprentissage
accéléré. La suite doit garder cette limite et déclarer séparément son expérience.
