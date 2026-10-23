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
21. **Rejeu de vérification (trouvé au verdict A1, Q\*bert)** : 5 des 6 vérifications Q\*bert d'A1
    divergeaient (88 autres, sur 22 jeux, concordaient). Première hypothèse, fausse : un état sauvé à
    l'image 0 se restaurerait mal ; la correction « ancre » qui en découlait (commits b1ad9ab, e849778)
    échouait encore sur les vrais runs et a été annulée (e696ffb, 7e10f35). Cause établie : juste après
    une remise à zéro, ALE garde un contexte hors de l'état sauvé qui modifie les images suivantes de
    Q\*bert ; l'émulateur vivant a toujours joué avant de restaurer quoi que ce soit, l'émulateur neuf
    de vérification non. Correction côté évaluateur seulement : l'émulateur de vérification joue un pas
    NOOP avant de restaurer la base. Avec elle, les 6 vérifications Q\*bert passent sur les fichiers A1
    d'origine (sans relance), les autres jeux restent justes, et deux relances complètes de Q\*bert
    reproduisent A1 à l'identique (mêmes enregistrements, mêmes images facturées : runs déterministes).
    Le verdict calculé avec l'ancienne vérification est gardé tel quel
    (`room_verdict_first_pass.json`, « Interruption ») ; le verdict officiel est recalculé avec la
    vérification corrigée, sans changer aucune règle.
22. **Après le verdict A1 (revue `wf_420b5f0a-bc1`)** :
    - Les relances Q\*bert (`qbert_check/`) ont tourné sur les commits annulés e849778 (9131) et b1ad9ab
      (9132). L'explorateur est inchangé depuis 2b22e94, mais le format npz est différent
      (`rec_start`/`start_bytes`, sans `base_state`) et la vérification actuelle ne sait pas le rejouer.
      Enregistrements et images facturées sont identiques à A1.
    - `room_verdict_addendum.json` est la sortie de la règle de levée déclarée puis annulée :
      `interruption_lifted: false`, 5 des 6 rejeux ancrés en écart. Négatif gardé, ce n'est pas le verdict.
    - σ̂² est poolé sur les 6 jeux à place de développement. Avec breakout et ms_pacman : σ̂² = 1,002,
      se_proj = 0,450, même verdict.
    - (d) est la médiane de deux valeurs : bank_heist (64 176 / 1 343 120) et jamesbond
      (1 561 072 / 283 072) passent grâce à une seule graine. Les 12 hausses d'ice_hockey 9132
      comprennent l'entrée, comme le prévoit la règle.
    - Barres négatives (fishing_derby, pong) : M1 = 0 est atteint par le clone racine, la vérification
      k = 1 rejoue zéro action, et M3 = 3|B| dépasse le score maximal.
    - Non produit : justesse de « moi » contre les juges RAM `*_v70` sur la calibration (synthèse §2.5) ;
      reporté, v83 s'arrêtant à A1.
    - Anomalies : gopher 9131 et name_this_game 9132 font environ 4 100 déroulés complets sans aucune
      hausse ; zaxxon fait 0/0.
    - Arrêts du superviseur pendant A1 : 24 (sortie 3), 21 tâches distinctes, toutes graine 9132, toutes
      relancées et terminées ; copie des lignes dans `craft_v6_out/v83/a1_supervisor_actions.txt`.
    - La provenance de `room_verdict.json` est tamponnée en fin de calcul, contrairement à la note 12.
