# Construction matérielle fixe, instrument CPU de développement

Convention distincte fixée au commit `108a59e`, avant ses contrôles : un objet
ALE neuf à chaque épisode, ROM et binaire épinglés, `system_random_seed=4753849`.
Le RNG sticky Python et le compteur global persistent ; le dernier bouton
revient à NOOP et les deux images appartiennent au nouvel épisode. Les bras
futurs devront partager cette convention avant leur gel. Elle change la
distribution des débuts d'épisode par rapport au reset naturel historique.

Le moteur est séparé : `ragnarok/environments/cold_emu_pool_v70.py`. Chaque
chargement facture ses 60 frames de détection plus 72 de reset (68 pour
Seaquest), soit 132/128, avant toute action. La construction et la répétition
demandée sont pré-admises ensemble. Les ROMs, options et empreinte du binaire
définissent la trace déterministe vérifiée k=0. L'API pixels n'expose pas RAM,
vies, score, terminal ni hook du juge. L'ancien moteur reste historique.

## Contrôles

`craft_v6_out/v70_cold_start_certification.json` : six jeux, deux seeds d'actions,
trois constructions par seed, 64 frames de jeu par construction. État initial
RAM/RGB identique, trajectoires répétées identiques et coûts exacts sur tous les
jeux. Total 7 032 frames facturées, sans apprentissage. Le seed matériel est commun
et fixe ; il ne s'agit pas de répétitions matérielles indépendantes.

`craft_v6_out/v70_cold_replay.json` : après 1 280 frames ordinaires par jeu, capture
d'une ancre explicitement non terminale, témoin de 32 actes, construction d'un
ALE neuf puis restauration. Les 32 états suivants sont identiques byte à byte
en RGB brut, RGB max-poolé et RAM ; aucune création supplémentaire lors de la
comparaison. Toutes les constructions et la restauration restent facturées.
Les jetons peuvent ainsi rester valides entre ces instances pour le développement.

La revue source signale que certains tampons vidéo TIA ne sont pas sérialisés.
Ces contrôles ne prouvent pas l'identité sur tous les états possibles : la
reproductibilité reste un contrôle obligatoire des futurs points de branche.
Le retour immédiat provient des deux RGB sauvegardés par le wrapper.

39 tests du transport historique et du moteur froid passent en 12,10 s :
refus avant construction, budget du repeat entier, sticky continu, comptabilité
des épisodes, sessions expérimentales et hook privé du juge après recréation.
Aucun gate, score de compétence ni apprentissage ARC 4 n'est revendiqué.

Sources de la justification : [construction ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/emucore/OSystem.cxx#L84-L103),
[RNG système](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/emucore/System.cxx#L34-L43),
[initialisation matérielle](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/emucore/M6532.cxx#L35-L46).
Les corrections des anciens comptes restent dans `docs/ARC4_RESET_AUDIT.md`.
