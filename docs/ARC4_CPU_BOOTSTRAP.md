# ARC 4 — premier module CPU, 8 septembre 2026

Ce jalon construit l'instrument d'émulation. Il ne constitue ni le gate 0, ni un
agent ayant appris la perception ou le transfert. L'ARC 3 conserve le GPU et son
environnement Python ; ce travail utilise un environnement CPU séparé, sans Torch.

## Exécution reproductible

Depuis la racine du dépôt, sous PowerShell :

```powershell
py -3.14 -m venv .venv-arc4
.venv-arc4/Scripts/python.exe -m pip install --only-binary=:all: -r requirements-arc4-cpu.txt
.venv-arc4/Scripts/python.exe -m pytest -q tests/test_emu_pool_v70.py
.venv-arc4/Scripts/python.exe -m scripts.check_emu_v70 --workers 2 --steps 512 --out craft_v6_out/v70_cpu_instrument_new.json
```

Le dernier chemin doit être nouveau : le script refuse d'écraser un résultat.
Versions directes fixées : ALE 0.12.1, NumPy 2.4.4 et pytest 9.0.3. La distribution
ALE fournit les ROMs ([notes officielles](https://ale.farama.org/release_notes/index.html)).
Seul **Pong, mode 0**, est activé ici. L'empreinte MD5 de la ROM auditée est
`60e0ea3cbe0913d39803477945e9e5ec`. Les autres jeux nécessitent une vérification
de leur coût de démarrage avant activation ; aucun jeu de confirmation n'a été ouvert.

## Interface et budgets

- `ragnarok/environments/emu_pool.py` lance des processus CPU isolés. `observe`
  et `step` renvoient des pixels RGB uint8, avec maximum des deux dernières images
  ou image brute à la demande. Récompenses, RAM, vies et indicateurs de fin restent
  dans le processus. Le compteur est une interface séparée pour l'expérimentateur.
- Les sauvegardes restent dans le processus ; l'appelant ne reçoit qu'un jeton
  opaque. Le mode `live` interdit sauvegardes et branches. Le mode `lab` permet
  une paire de branches revenant au même point de départ.
- Un `clone` explicite ouvre une session de laboratoire : tant qu'un jeton existe,
  les pas et redémarrages sont expérimentaux. Cela empêche de faire passer les
  essais restaurés pour du jeu ordinaire. L'archive persistante E4 demandera une
  interface distincte, encore à construire.
- `frame_budget_v70.py` conserve un compteur global qui ne recule jamais, y compris
  après restauration. Il sépare frames ordinaires, frames expérimentales, charges
  de restauration et décisions de l'agent. Une restauration coûte 60 frames.
- Une paire de deux branches de quatre frames coûte 128 frames, dont 120 de
  restauration. Avec une part expérimentale de 20 %, elle demande au moins 512
  frames ordinaires disponibles ; à 5 %, 2 432. Le démarrage réel entre dans ces
  comptes. L'admission précède toute action et couvre les deux restaurations.
- La part expérimentale est fixe par instance (20 % par défaut). Le calendrier
  de décroissance du protocole reste à définir avant gel ; il n'est pas inventé
  par ce module. Le plafond du pool est de 100 000 frames facturées par défaut.
- Une branche s'arrête à la fin d'épisode. Le prochain `step` admet et facture
  le redémarrage avant de jouer. Les limites et comptes sont propres à chaque
  processus ; un lot n'est pas atomique entre processus. Après une erreur logique,
  toutes les réponses sont consommées ; une rupture de transport ou interruption
  ferme le pool pour éviter de réutiliser une réponse ancienne.

## Trois pièges ALE vérifiés

La revue indépendante du code ALE 0.12.1 et les essais sur Pong ont identifié :

1. L'état natif, même avec `include_rng=True`, ne suffit pas à reproduire les
   actions persistantes : leur historique n'est pas dans cet état. Le module
   désactive donc les actions persistantes natives et possède son RNG, sa dernière
   action et une probabilité par frame de 0,25. Il sauvegarde ces éléments avec
   l'état natif. Cette convention ne reproduit pas le flux aléatoire natif d'ALE.
   Sources : [état sérialisé](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/environment/ale_state.cpp)
   et [état de l'environnement](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/environment/stella_environment.hpp).
2. Le tampon RGB peut rester celui du futur abandonné juste après restauration.
   Les deux images brutes sont donc copiées dans la sauvegarde et restaurées sans
   jouer de frame supplémentaire. Source : [interface ALE](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/ale_interface.cpp).
3. Les getters de frames ne comptent pas le démarrage interne. Pour cette ROM et
   ce mode, chargement et reset exécutent 72 frames : 60 NOOP, puis trois blocs
   de quatre frames. Ce coût est facturé explicitement. Les frames de jeu suivantes
   sont exécutées une par une et vérifiées par delta réel ; une répétition tronquée
   par une fin d'épisode ne reçoit pas quatre frames fictives.
   Sources : [reset ALE](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/environment/stella_environment.cpp)
   et [mode Pong](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/games/supported/Pong.cpp).

## Validation du jalon

Les **27 tests passent en 9,91 s** : limites exactes, comptage monotone, images
brutes et agrégées, branches reproductibles, trajectoire ordinaire préservée,
fins d'épisodes et resets, sessions explicites, erreurs partielles de lot,
interruption et fermeture des processus appartenant au pool.

La revue contradictoire a fait corriger le classement des pas après `clone`,
le traitement des envois partiels, des interruptions de réception et d'un échec
de création de processus. Ces corrections sont couvertes par les tests.

Résultat final : [v70_cpu_instrument_verified.json](../craft_v6_out/v70_cpu_instrument_verified.json).
Deux processus, 512 décisions chacun : environ **1 144 décisions/s**, soit
**4 575 frames ordinaires/s**, pendant que l'ARC 3 tournait. Le chronométrage porte
sur les appels ordinaires du pool, hors démarrage, perception, évaluateur et
apprentissage. Ce débit court ne valide pas le débit complet du futur gate.
La paire ajoute exactement huit frames et deux restaurations ; les trois contrôles
de reproduction et de trajectoire passent. Le JSON conserve versions et empreintes
des sources, avec `gate_result: null`.

Le premier essai `v70_cpu_instrument.json`, antérieur aux dernières corrections
de revue, est conservé avec ses propres empreintes ; le fichier `verified` est le
résultat associé au code final de ce jalon.

## Suite

Calibrer indépendamment l'évaluateur externe sur des fixtures, construire
l'extracteur de référence et mesurer le débit de la chaîne complète. Résoudre
ensuite les jeux/graines, le budget d'entrée et les conventions encore ouvertes
avant de geler le gate 0. La mesure distinguant adaptation accélérée et avance
initiale reste une décision scientifique ouverte pour les étapes suivantes.
