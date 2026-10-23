# v83 — notes de construction (écarts d'implémentation, aucun critère touché)

Ce fichier liste les choix d'implémentation qui diffèrent de la disposition prévue par la synthèse
(`docs/panels/v83_promise_map_synthesis.md` §5). Il ne modifie ni les critères, ni les seuils, ni les
graines, ni les bras du protocole figé (`docs/ARC4_V83_PROTOCOL.md`).

1. **Sens hors du paquet agent.** `SenseEnv`, `Trace` et `Spent` sont dans
   `ragnarok/evaluation/senses_v83.py` et non dans `ragnarok/agent_v83/senses.py` : la classe tient
   l'émulateur et le score vrai ; la placer côté évaluateur rend l'isolation plus stricte (le paquet
   agent n'importe rien de `ragnarok.evaluation`, ni `ale_py`, et ne touche aucun attribut privé des
   objets reçus ; test AST `tests/test_isolation_v83.py`).
2. **Trace.** Elle n'enregistre que les clones qui battent le meilleur score vrai cumulé (le premier
   clone atteignant un palier M en fait forcément partie) et garde la lignée de chaque poignée (parent,
   boutons depuis) pour rejouer depuis l'état de remise à zéro, hors registre facturé.
3. **`reset()` et `restore()` rendent `(obs, vies)`**, pour que la case de la racine porte ses vies.
4. **Adaptateur.** Copie de `agent_v71/adapter.py` avec fabrique (`env.sibling(991)`), pas étape à
   4 éléments, poignées libérées, sans noms de boutons (l'agent ne voit que des indices).
5. **Fond.** Mode par pixel sur les 12 000 pas d'échauffement, chaque image (l'échauffement v74 mettait
   aussi à jour le fond à chaque image avant la recherche de compteurs).

## Corrections de la revue d'avant lancement (workflow `wf_015d274f-fc1`, 22 octobre, avant toute image A1)

6. **Étiquettes de lignes** : une ligne n'est gardée que si sa fenêtre [t, t+H) tient dans le déroulé
   prévu (t + H ≤ 100) et a été vue en entier, ou si la baisse de vies a elle-même arrêté le déroulé
   (MAUVAIS) ; jamais sélectionnée sur son issue (l'ancienne règle gardait les fins de déroulé
   seulement quand l'événement arrivait : biais vers la réussite de G-préd). Champs `t` et `miss`
   stockés.
7. **« Moi perdu »** = pas de position ou raté du traqueur au pas courant (comme le contrôle d'entrée) ;
   fenêtre alors centrée sur l'écran, rien de masqué ; `lost_steps` compté ainsi ; la clé de case garde
   la dernière position du traqueur (comportement v74).
8. **Masque de moi** : composante sous moi, sinon la plus proche à ≤ 4 px ; d'autres composantes de la
   boîte seulement si le total reste ≤ 1,5× ma taille ; ma composante précédente retirée du canal
   « disparus ».
9. **Taille de moi pour la vue** = médiane d'entrée, fixe pour le run (la sonde réécrit la taille du
   traqueur).
10. **Crochets** `on_put(ex, case, ctx)` et `on_iter(ex, it)`, appelés seulement si un pilotage existe ;
    V inchangé (test : un pilote qui tire comme V donne exactement les mêmes lignes, itérations et cases).
11. **Verdict A1** : plancher C ≥ 5 000 appliqué ; écart de rejeu, code mêlé ou sale = Interruption ;
    sans rejeu = « unconfirmed », écrit hors de `room_verdict.json` ; un run non conforme (graine,
    plafond, arrêt anticipé, pilotage) = incomplet. Pas u : si les épisodes de référence n'ont aucune
    hausse, u = max(0,5·|B|, médiane des hausses des runs), sinon 0,5·|B|.
12. **Provenance** tamponnée au lancement de chaque tâche ; A1 tourne depuis un worktree propre dédié
    (`F:\dev\ragnarok_v83run`, figé au commit A1) pendant que le code des promesses avance ailleurs.
13. **Cône** : distances mesurées depuis ma position dans la fenêtre orientée (le centre exact dépend de
    la rotation d'une fenêtre paire).
14. **Hausses pendant les sondes** : les branches de sonde sont des images facturées où l'agent sent les
    hausses ; elles comptent dans le critère (a) (« hausses senties »). Rejeté comme défaut par la revue,
    noté ici par transparence.
15. **Unités d'A2/A3** : les jeux à place parmi les 41 unités d'A1 (développement et calibration) ; le
    seuil G-place ne compte que les 35 jeux de développement (inchangé).
16. **Entraînement de W** (déclaré avant tout résultat A1, `scripts/train_promise_v83.py`) : lignes
    « perdu » retirées ; au plus 15 000 lignes par run tirées uniformément (graine 9149) ; poids égal
    par jeu ; ordonnées libres par jeu ; crête 1e-4 ; L-BFGS ≤ 300 itérations ; placebo = décalage
    circulaire des étiquettes de s lignes, s uniforme entre les lignes de 10 déroulés et n moins
    celles-ci, puis normes égalisées.
17. **A3 « le meilleur de V et F »** : pris jeu par jeu (le bras comparateur au plus petit log C moyen
    sur ce jeu, et de même dans chaque rééchantillon) ; drapeau KC-nuisance = rapport poolé des montées
    F/V > 1,25 (censure de comparateur pour les deux). Déclaré avant toute image A3.
18. **W₋g et W_perm,₋g** ajustés depuis zéro (aucun départ à chaud depuis un W qui a vu g) ; une seule
    standardisation par ensemble d'entraînement, commune aux deux têtes.
19. **Revue du code des promesses (`wf_fd7ce591-9d8`, avant A2 et toute image F, X, P, X0, K\*)** :
    une tête transportée sans positif garde w, son ordonnée et son pilotage ; MAUVAIS à ordonnée seule
    tant qu'aucune baisse de vies n'est sentie ; tête neuve à ordonnée seule ; F recalcule sa
    standardisation à chaque réajustement ; toute échelle ≥ 0,05 (F et W) ; bouton de début de déroulé
    noté sur la vue archivée de la case (la racine garde une vue sans mouvement) ; placebo décalé dans les
    seules lignes non censurées de chaque tête, jeu à ≤ 20 déroulés gardés écarté des ajustements poolés
    de cette tête ; A2b : n premières lignes non perdues ; A3/A4 : rejeu des paliers 1, 2, M\* (écart =
    Interruption, sans rejeu = « unconfirmed »), code comparé par arbre (`code_tree`), W et têtes hachés
    par run ; p de permutation recalculé sous l'étiquette permutée, (1+b)/(1+n) ; drapeau KC-nuisance
    calculé avec la censure du bras transporté pour F (un F piégé ne paraît jamais rapide) ; seuil mémoire
    14 Go pour les files A3/A4 (un réajustement tient ~0,6 Go de plus).
20. **Script de verdict A1** : une correction de revue (repli du pas u) avait laissé une erreur de
    syntaxe dans `scripts/room_verdict_v83.py` (commit A1 `2b22e94`) ; trouvée par un essai à blanc sur
    des données partielles, corrigée sans changer la règle. Le worktree de mesure n'est pas touché (les
    tâches A1 y tournent encore) : le verdict est calculé dans `ragnarok_v83` sur les sorties importées
    par `scripts/a1_import_v83.py` (copie vérifiée octet par octet, manifeste SHA-256), en vérifiant que
    tous les runs A1 portent le même code (`2b22e94`, propre).
21. **Rejeu ancré (trouvé au verdict A1, Q\*bert)** : sur Q\*bert, un état sauvegardé à l'image 0
    d'épisode (juste après la remise à zéro) ne se restaure pas fidèlement (7 octets de RAM diffèrent
    après un pas, avec ou sans le générateur aléatoire de l'émulateur ; testé : jamais au-delà de
    l'image 0). Le rejeu de vérification partait de cet état de base, alors que le run vivant repart
    d'états restaurés : 5 des 6 vérifications Q\*bert d'A1 divergent, les 88 autres (22 jeux)
    concordent. Correction côté évaluateur seulement (le comportement de l'agent ne change pas) : la
    trace garde l'état du premier clone pris après l'image 0 sur chaque lignée (« ancre ») et le rejeu
    part de là ; test de régression `test_replay_exact_where_frame0_restore_is_not` (6/40 écarts avec
    l'ancien rejeu, 0 avec l'ancré). Le verdict A1 enregistré reste « Interruption » ; la levée de
    l'Interruption passe par une ré-exécution des deux runs Q\*bert avec ce rejeu (addendum).
