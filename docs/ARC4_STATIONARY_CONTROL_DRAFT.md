# Contrôle positif stationnaire : construction et certificat avant tout fit

10 septembre 2026. Cette nouvelle calibration suit les cinq fits synthétiques
de mouvement et le premier fit Atari, tous négatifs à leurs critères globaux.
Elle ne relance aucun de ces essais, ne change aucun seuil et n’ouvre aucun
jeu. Le raccord P→m→C/ME et les nouveaux fits Atari restent suspendus.

## Question distincte

La reconstruction par déplacement admet des solutions non physiques. Le fit
Atari est négatif, mais sa courbe ne démontre pas la convergence. Avant de
modifier encore une architecture ou de collecter plus d’images, préparer une
tâche dont **le réseau exact possède une solution finie connue**. Un éventuel
fit séparé pourrait alors tester l’ajustement dans un budget fixe, sans confondre
son échec avec une impossibilité de représentation. Ce document n’autorise
encore que la construction et la certification du contrôle, pas son fit.

## Monde et observations

Douze écrans RGB 210×160. Dix ont respectivement 1/4/16/64/128/256/336/32/8/2
pixels rouges ; deux, aux positions 6 et 7, sont vides. Ce sont douze cas d’un
monde synthétique, pas six nouveaux jeux. Les noms `synthetic_pair_0` à `_5`
servent seulement au regroupement des mesures par paires, sans entrée du réseau.
Positions tirées sans remise, avec un coin imposé selon l’indice du cas,
les indices des deux cas vides étant inclus dans cette rotation.
NumPy PCG64, graine de données 774, fixée avant génération. Le rouge est binaire
0/255 ; vert arbitraire ; bleu initial nul aux pixels rouges, arbitraire ailleurs.

La loi du simulateur est locale et stationnaire : NOOP conserve les RGB ; la
commande 1 met le bleu à255 partout où le rouge vaut255 et le laisse ainsi.
Deux pas NOOP donnent les deux passés. Depuis le même état possédé, chaque
commande est ensuite tenue huit pas dans sa propre branche. La cible est
UNIQUEMENT l’union des différences RGB entre ces deux branches. Aucun masque
du générateur n’est retourné ou utilisé comme cible. Les tests suppriment la
réponse du simulateur et vérifient que les labels disparaissent malgré le rouge.

Les 216 pas et images RGB synthétiques (18 par cas) sont déclarés séparément
des images émulées Atari, dont aucune n’est produite ici. Les quatre tableaux
conservés sont passés, futurs NOOP, futurs commande1, enveloppes observées.
Un futur learner ne recevrait comme entrée que les deux passés, et comme
supervision les différences de ses branches ; aucune loi de couleur, coordonnée,
commande, poids de référence ou métadonnée du générateur en entrée.
Les douze écrans initiaux générés sont comptés séparément des216 retours de pas.
Un ledger externe persiste chez l’appelant : chaque entrée de pas et chaque
retour RGB y sont comptés séparément, même si le pas ou l’écriture du journal
suivant échoue. Les retours sont journalisés ; une entrée sans retour reste
explicitement incertaine. Les cas initiaux complets sont comptés un par un.

Cette famille est volontairement facile : une couleur visible indique une
réponse stationnaire déterministe. Elle ne représente ni sprites en mouvement,
ni collisions, ni occultations, ni contrôle d’un jeu réel.

## Certificat de référence, exclusivement dans le juge

`ResponseEnvelope` reste inchangé : cinq conv3×3 à16 canaux, dilatations
1/2/4/8/16, padding répliqué, SiLU, tête1×1. Ses10177 paramètres sont conservés.
Tous les poids/biais de référence sont nuls sauf les coefficients centraux :

| Couche | Sortie0 | Sortie1 |
|---|---|---|
| Première | +4 sur chacun des deux R ; biais−4 | −4 sur chacun des deux R ; biais+4 |
| Quatre suivantes | +1 canal0, −1 canal1 | −1 canal0, +1 canal1 |
| Tête, unique sortie | +2 canal0, −2 canal1 | — |

Dans cette preuve, `R` désigne le rouge normalisé par255, donc0 ou1.
Si `phi=SiLU`, alors `phi(t)-phi(-t)=t`. Avec
`t=4*(R_passé1+R_passé2)-4`, chaque couche conserve la paire `phi(t),phi(-t)`.
La tête vaut `8*(R_passé1+R_passé2)-8`, soit +8 sur les rouges et−8 ailleurs.
Les poids hors centre n’interviennent pas : preuve valable aux bords et coins.

Au seuil inchangé0,5 : IoU/AP1 sur les cibles non vides, zéro faux positif sur
les vides ; BCE équilibrée `log(1+exp(-8))`, environ0,0003354. Les vides gardent
une IoU indisponible. La preuve vaut pour toute la famille décrite, pas seulement
les douze tirages. Une vérification numérique de cette référence est aussi
requise : logits à1e-5 de±8, BCE à1e-6 de la formule, décisions exactes.

Cette fabrique et ses poids restent dans `ragnarok/evaluation`. Aucun transfert,
initialisation du learner, distillation ou cible de features depuis cette
référence. Un éventuel learner sera un autre processus, initialisé séparément.
Le certificat ne garantit pas une avance sur son initialisation, ni une
convergence. Un témoin initial déjà parfait devra être publié sans fabriquer
après coup une famille plus difficile.

## Exécution de construction/certification uniquement

Une tentative exclusive, CPU un thread, CUDA vide. Générer exactement ce lot,
enregistrer les quatre tableaux et les poids de référence, effectuer un seul
forward de juge sur le lot12, publier logits et mesures. Zéro optimizer.step.
Les petits forwards des tests synthétiques se déclarent séparément ; aucun
NPZ réel antérieur n’est lu pendant ces tests ou cette construction.

Plafond300s, réserve10s incluse, au plus64MB de sorties binaires. Imports et
preuves initiales exclus ; génération, calcul de référence, sérialisation,
fsync/hash, journal fermé et nouvelles vérifications de provenance inclus.
Seul le petit reçu final et la console suivent la mesure. Une erreur, un
dépassement ou certificat numérique manqué donne STOP ; aucune seconde
construction, modification de graine ou extension. Toutes les traces restent.

Avant lancement : sources, protocole, tests et revue indépendante committés.
Après : préserver, faire revoir la provenance et le certificat, puis décider
techniquement d’un éventuel essai d’ajustement déclaré séparément. Aucun succès
de ce contrôle ne réhabilitera les résultats Atari ou ne prouvera le transfert.
