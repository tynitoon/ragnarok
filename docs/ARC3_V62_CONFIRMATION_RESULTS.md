# ARC 3 v62 : confirmation complète, résultat non concluant

10 septembre 2026. Les neuf mondes et les trois lignées ont terminé leur
budget fixé de six tours par objectif. La revue indépendante est conforme.
Le scorer gelé donne **INCONCLUSIVE** : un petit avantage moyen de
l’expérience passée est présent, mais il ne se retrouve pas dans chacune
des trois lignées. L’apprentissage plus rapide n’est donc pas démontré par
cette confirmation. Aucun monde ni critère n’est remplacé après ce résultat.

## Résultat et motif exact

La mesure principale est la moyenne des performances après les tours1 à6,
sur les trois objectifs de chaque monde, puis sur les neuf mondes. Elle
compare M4, entraîné auparavant dans quatre mondes, à la moyenne de deux
agents frais qui apprennent eux aussi pendant la confirmation.

| Mesure gelée | Valeur |
|---|---:|
| Δ principal M4 − frais | +0,0549321 |
| Erreur standard retenue | 0,0263868 |
| Seuil2×erreur standard | 0,0527736 |
| Erreur standard du témoin frais–frais | 0,0263868 |
| Plancher de résolution | 0,0069444 |
| Erreur standard de robustesse, descriptive | 0,0646530 |
| Ligne instrumentale `t_init` | −0,8038663 |
| Aire complète0..6, descriptive | +0,0452243 |

Le seuil global2×erreur standard est franchi de peu. **La condition qui
échoue est l’avantage strictement positif dans chaque lignée** :

| Lignée, trois mondes chacune | Avantage moyen | Erreur standard descriptive |
|---|---:|---:|
| 0, mondes8300–8302 | −0,0446972 | 0,0651307 |
| 1, mondes8310–8312 | +0,2106509 | 0,0299532 |
| 2, mondes8320–8322 | −0,0011574 | 0,1623597 |

Le scorer conserve son message générique, qui énumère avec « or » le seuil
global et la condition de lignée. Dans ce cas, sa partie numérique
`0.0549 < 0.0528` est fausse ; c’est la seconde condition qui déclenche le
verdict. Sa phrase gelée « not distinguishable from zero at this N » reste
archivée, mais ne décrit pas précisément ce déclencheur. Le code, les
nombres, le verdict et la figure n’ont pas été réécrits après observation.
`t_init`, malgré son nom historique, porte sur l’écart des aires des deux
agents frais ; ce n’est pas seulement leur différence au départ.

## Où les acquis aident et où ils coûtent

| Objectif | M4, aire1..6 | Frais, aire1..6 | Δ | Erreur standard | Écart au départ | Écart final |
|---|---:|---:|---:|---:|---:|---:|
| Niveau2, premier objectif | 0,885996 | 0,990308 | −0,104312 | 0,012028 | −0,127606 | −0,012156 |
| Niveau3, deuxième objectif | 0,580157 | 0,472509 | +0,107648 | 0,074552 | +0,039067 | +0,080750 |
| Niveau4, troisième objectif | 0,239872 | 0,078412 | +0,161460 | 0,015096 | +0,049472 | +0,226556 |

L’agent expérimenté fait mieux sur les objectifs plus difficiles, surtout
le troisième, mais moins bien sur le premier objectif facile. Ces nombres
décrivent les neuf mondes déjà fixés ; ils ne constituent pas de nouveaux
tests choisis pour remplacer le résultat principal. Le niveau3 ne franchit
pas son seuil de formulation « learns faster » :0,107648<0,149103.

Le store reste conservé entre objectifs d’un même monde, pour tous les bras.
L’écart au départ des niveaux3/4 contient donc **poids et mémoire du monde**.
Au premier objectif, il ne contient que les poids transférés. L’aire après
le départ peut elle aussi conserver l’effet d’une avance initiale : l’exclure
du premier point ne suffit pas à isoler une vitesse d’apprentissage.

Le probe préalable est positif :618/768 premières propositions conformes aux
lois pour M4, contre398/768 pour l’initial aléatoire. Cela fournit une preuve
d’acquis réutilisés dans ce probe. Il n’y a pas d’AUC de la tête de loi mesurée
sur les cellules exercées en confirmation ; l’AUC du témoin « roster-only »
ne doit pas être attribuée à l’agent appris. Le graphique M1..M4 reste
descriptif : effectifs inégaux et expérience confondue avec nombre de gradients.

![Courbes complètes et probe](F:/dev/ragnarok/craft_v6_out/v61_figure.png)

La figure conserve le texte du scorer gelé ; le motif de lignée et la limite
sur la vitesse sont explicités ci-dessus. La courbe G’ ne porte que sur les
deux mondes annoncés de la lignée0 ; L est une référence qui connaît les lois.

## Provenance, coût et clôture

v62 est une refonte après le STOP de faisabilité de v61, déjà déclarée avant
son gate sur8102/8103. Le protocole autoritaire est ARC3_PLAN.md§12.6 et§13.
B6, neuf mondes, trois lignées,64environnements et G’ sur deux unités restent
inchangés. Deux interruptions techniques ont été conservées et comptées.

La confirmation s’est arrêtée le10 septembre à05:50:34,294716UTC, avant
l’échéance originale09:10:02UTC. Le coût compté jusqu’à sa clôture est
139232,2044323secondes, soit **38,6756123heures sur42**, pannes et attente
comprises. Avec arrondi conservateur et toute la préparation murale de la
revue finale,140257,9097163secondes restent sous le plafond. Ces durées sont
des coûts budgétaires conservateurs, pas un relevé du seul temps GPU actif.

Résultats finaux scellés6c7f677 ; revue indépendante3cab686 :17sources,
133preuves,41artefacts des lignées vérifiés sans décodage de poids ni inférence.
Les poids historiques sont liés aux reçus committés ; certains anciens logs
auxiliaires sont seulement hachés par cette revue, distinction conservée dans
son reçu. Le scorer gelé a été exécuté une fois après PASS ; la figure corrigée
423b770, préalablement revueaf94a62, a été produite une fois et inspectée.

- Agrégat : `craft_v6_out/v61_confirm.json`.
- Revue : `craft_v6_out/v62_confirm_final_review.json`.
- Sortie exacte du scorer : `craft_v6_out/v61_confirm_score.json`.
- Courbes : `craft_v6_out/v61_figure.png`.
- Reçu de clôture : `craft_v6_out/v62_confirm_publication.json`.

ARC3 est clos avec ce verdict. Il n’y a ni relance, ajout de monde, prolongation,
sélection de lignée favorable ni changement de seuil. ARC4 poursuit le passage
aux pixels et boutons dans ses propres protocoles. Ces mondes symboliques ne
démontrent ni maîtrise Atari/NES, ni transfert entre vrais jeux.
