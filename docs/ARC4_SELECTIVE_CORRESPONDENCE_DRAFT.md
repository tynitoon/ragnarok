# ARC 4 — correspondance sélective : mesure avant raccord

Proposition prospective du 10 septembre 2026. **Aucune donnée de cette nouvelle
famille n’a été générée et aucun apprentissage n’est lancé.** Ce texte fixe le
contrat de mesure et son instrument. Le générateur, l’architecture exacte et
le budget d’un futur apprentissage devront être spécifiés et relus séparément
avant qu’un essai soit exécutable. Les anciennes calibrations restent closes.

## Question et domaine de validité

La prochaine épreuve demande si une association apprise depuis les observations
peut fournir assez de correspondances correctes tout en s’abstenant dans les
situations où elle se tromperait. Elle ne certifie pas une identité physique
pour tous les mondes compatibles avec les pixels : les deux témoins d’occultation
conservés démontrent que cette garantie absolue est impossible en général.

La portée sera statistique et limitée à une famille procédurale déclarée.
La correspondance sera construite après réception des images ; aucun futur
n’entrera dans la prédiction préalable. Aucun label d’objet, de mouvement,
d’occultation ou de contrôle du juge ne servira à apprendre ou régler l’admission.
La politique ne reçoit ni boutons ni codes locaux non alignés aux douze effets.

La piste étudiée apprend des représentations dont les associations temporelles
reviennent au point source dans une séquence parcourue puis inversée. C’est le
principe des marches contrastives de Jabri et al. ; leur article signale aussi
des raccourcis possibles lors d’un apprentissage sur une carte dense unique.
La reprise envisagée sera une adaptation nouvelle, pas une reproduction de
leurs résultats. [Article, §2 et annexe C](https://arxiv.org/html/2006.14613v2).
L’utilité d’un encodeur limité aux couleurs est encore en revue : retrouver
une couleur inchangée ne suffirait pas à démontrer une correspondance apprise.

## Unité de mesure et séparation du juge

Cinq strates de **256 scènes tirées i.i.d. dans une loi fixée par strate** : mouvement visible,
immobilité visible, occultation finale, sortie d’écran finale et ambiguïté
d’apparences identiques. Les deux premières ont une correspondance vraie unique
dans la famille. Les trois autres exigent l’abstention sur la requête évaluée.
Les textures, dispositions et tirages d’évaluation seront séparés des tirages
d’apprentissage dès la déclaration du générateur. Aucun tri après score.

Dans chaque scène, le juge choisit exactement **une requête source** selon une
règle prospective indépendante du candidat. Le modèle produit d’abord ses
associations pour tous les pixels, sans recevoir la requête ni le masque vrai.
Le juge extrait ensuite la réponse à sa requête. Ainsi les 64 pixels voisins
d’une même image ne deviennent pas 64 expériences indépendantes. La règle de
choix du point, l’unicité de sa correspondance et le générateur restent des
éléments à auditer avant collecte ; le score seul ne peut les certifier.

Une prédiction gelée du modèle initial et une prédiction finale du modèle appris
sont appariées sur chaque même scène. Aucun ajustement ou seuil n’utilise les
labels d’évaluation. Toutes les scènes, admissions, erreurs et omissions sont
conservées, y compris lorsque le candidat s’abstient partout.
Les deux modèles et leurs règles d’admission restent fixes durant l’ensemble de
l’évaluation. La borne du risque conditionnel reste valide conditionnellement
au nombre aléatoire d’admissions, sous cette hypothèse i.i.d. par strate.
Les admissions sont enregistrées avant extraction du verdict du juge. Ni strate,
ni requête, ni masque vrai ne doivent pouvoir changer une admission en abstention.
Les conclusions portent sur la population des requêtes choisies et sur les deux
modèles évalués : ni validité sur tous les pixels, ni variabilité entre entraînements
ne sont établies par ces comptes. Le futur reçu devra nommer cette population.

## Neuf exigences simultanées

Le risque global de fausse affirmation retenu pour cette épreuve est 0,05,
réparti par Bonferroni entre **neuf exigences**, soit alpha = 0,05/9 chacune.
Ce nombre est distinct du taux maximal d’erreur d’association, lui aussi fixé
à 0,05. Ces choix bornés sont des critères de développement, pas une tolérance
déjà validée pour injecter des pseudo-labels dans le reste de l’agent.

Pour chacune des deux strates visibles, trois exigences sont nécessaires :

1. Borne supérieure unilatérale exacte de l’erreur parmi les admissions ≤0,05.
   Le dénominateur est le nombre de scènes admises. Sans admission, la borne
   vaut 1 et ce critère échoue.
2. Borne inférieure unilatérale exacte de la proportion de correspondances
   correctes effectivement fournies, sur les 256 scènes, ≥0,50.
3. Gain observé de cette couverture correcte sur l’initial ≥0,10 **et** test
   binomial apparié unilatéral sur les paires discordantes, p ≤ alpha.
   Les succès sont « admis et correct » ; abstention et erreur sont des échecs.
   Sans discordance, p vaut 1. Ce test vise une amélioration sur zéro, avec
   un gain observé minimal ; il ne borne pas le gain vrai à au moins 0,10.

Pour chacune des trois strates non attribuables, la borne supérieure exacte du
taux d’admission indue, sur les 256 scènes, doit être ≤0,05. Toute admission y
est indue. Les risques ne sont pas noyés dans une moyenne de strates faciles.

Les bornes sont celles de Clopper–Pearson, unilatérales au niveau 1−alpha.
Le test apparié compte les gains parmi gains + pertes sous probabilité nulle
1/2. [Test binomial SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html),
[bornes exactes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
L’hypothèse i.i.d. requise porte sur les scènes au sein de chaque strate ; elle
reste une condition du générateur et du gel des prédictions, pas un résultat
produit par cette formule. Bonferroni ne requiert pas l’indépendance des critères.

Les quatre contrôles existants (deux contre-exemples d’occultation, chacun en
raw et pooled) sont obligatoires en plus : aucune attribution à l’objet masqué.
Leur conformité est un contrôle déterministe, pas quatre scènes aléatoires
supplémentaires dans les intervalles de confiance.

## Instrument et arrêt

L’instrument de score recevra uniquement les verdicts du juge et identifiants
de scène, jamais des images ou un modèle à relancer. Il refusera les scènes
dupliquées, les strates incomplètes, les types ambigus et les verdicts incohérents.
Un résultat inconnu n’est pas un succès et une couverture nulle ne peut pas passer.

Seules ses fixtures arithmétiques sont autorisées à ce stade : **60 secondes
cumulées CPU**, un thread numérique, zéro entraînement, image procédurale,
émulation ou archive expérimentale. Elles vérifieront les cas extrêmes des
bornes, les dénominateurs, l’absence de double comptage et les décisions sur
des verdicts inventés. Ce budget est distinct des 60 s de l’interface désormais
close. Une conformité ne constituera pas un résultat du candidat.

Un éventuel PASS du futur candidat autoriserait à proposer un raccord limité,
pas à activer automatiquement m/C. m exige sa comparaison propre de mouvement
et immobilité ; C exige la correspondance dans les deux branches d’un horizon
commun. La sélection de l’intersection des admissions peut concentrer leurs
erreurs : les risques mesurés séparément ne garantissent pas le risque conjoint
de C, ni celui de la comparaison à deux explications de m. Ces admissions
conjointes devront être évaluées explicitement avant leur raccord.
Un effet peut rester indirect et ne certifie pas ME. Aucun gate, jeu
réservé, validation conditionnelle ou résultat d’apprentissage accéléré n’est
ouvert par ce document.
