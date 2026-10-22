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
