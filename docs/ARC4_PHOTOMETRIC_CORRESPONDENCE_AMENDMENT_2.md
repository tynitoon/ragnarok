# Correspondance photométrique — clôture avant toute exécution

10 septembre 2026. Précision prospective issue de la revue du lanceur : aucune
fixture, donnée, initialisation ou mise à jour de ce candidat n’a encore été
exécutée. Les lois, graines, neuf critères, modèle et 512 mises à jour restent
ceux déjà déclarés. Les plafonds de calcul restent 60/300/60 secondes.

La mesure du temps inclut imports, génération, calculs, sauvegardes de tous les
payloads et reçus, fsync, relectures et empreintes. **Seules la publication atomique
ultime du marqueur de clôture, par lien dur sans écrasement, et la console sont
hors de cette durée scientifique.** Aucun calcul de données, réseau ou score ne
peut intervenir après cette publication. Cette frontière évite une régression
infinie où vérifier l’écriture du dernier reçu exigerait encore un autre reçu.

Le marqueur est préparé dans un fichier pending conservé, fsync, relu et haché.
Le lanceur vérifie ensuite le temps avec une seconde de marge et le stockage,
puis publie ces mêmes octets sans nouvelle lecture. Le consommateur exige ce
marqueur positif et son lien SHA au reçu : un JSON « complete » seul n’autorise
rien, et une erreur de clôture laisse le jalon non certifié.

Le plafond de 64 MB admet chaque payload avant écriture, fragments et fichiers
pending conservés compris. Une réserve de 65 536 octets protège la clôture.
Chaque stdout/stderr enfant est effectivement plafonné à 16 384 octets ; un
dépassement conserve le préfixe, arrête le processus et interdit le succès.

Chaque lot de réponses calculé est sauvegardé avant le suivant. Les étapes
d’apprentissage ont des événements durables de début, entrée d’optimiseur et
fin. En cas d’interruption, les compteurs d’entrée ne deviennent pas des retours :
les bornes du travail en vol restent explicites. Les tables de génération sont
conservées dès leur retour, avant le calcul suivant ; une table non retournée
est comptée par bornes, sans inventer son achèvement.

L’état initial, l’état final complet et les sorties partielles éventuelles sont
distincts. La génération et le fit sont deux processus ; l’empreinte du manifeste
de génération est épinglée entre eux. Le juge recontrôle observations scellées,
sources et versions de calcul, conserve ses entrées avant son unique score et
dispose de sa propre clôture positive. Toutes ces règles précèdent l’essai.
