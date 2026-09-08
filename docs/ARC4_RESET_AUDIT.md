# Audit des coûts de reset, développement seulement

Protocole fixé dans `b0d795b`, avant ouverture des ROMs de développement
supplémentaires. Le premier essai Breakout s'arrête après le chargement seed21000 :
le script supposait à tort une méthode Python `getMode`, absente d'ALE 0.12.1.
Ce chargement seul a consommé le démarrage nominal de 72 frames ; aucune frame
de jeu ni donnée de calibration n'a été collectée. L'essai échoué reste dans
l'historique, sans résultat de performance.

Le script corrigé conserve le mode par défaut défini dans les sources et relève
les octets utilisés par la sélection du mode, sans appeler `setMode` ni modifier
le protocole. Les octets après le soft-reset final sont diagnostiques : ils ne
doivent pas être confondus avec la condition lue avant ce reset.

Les 15 chargements/resets par jeu donnent tous k=0 ; les coûts du reset ALE
observés sont donc 68 pour Seaquest, 72 pour les quatre autres. Ce n'est pas
une borne démontrée pour tous les futurs états.

## Correction découverte par la revue approfondie

La console exécute **60 frames supplémentaires au premier chargement** pour
détecter NTSC/PAL, avant le reset ALE. Les logs des six ROMs indiquent bien
`AUTO-DETECT ==> NTSC`. Source primaire :
[Console ALE 0.12.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/emucore/Console.cxx#L137-L155).
Ces frames n'apparaissent pas dans les getters ALE et ne se répètent pas lors
des `reset_game` ultérieurs. Le chargement initial est donc 132 frames pour
Pong, 128 pour Seaquest, et 132 + c·k pour les quatre autres.

Les cinq JSON d'audit et les anciens artefacts Pong sont conservés tels quels :
ils omettent 60 frames par instance chargée. Le total de chaque audit à trois
seeds doit recevoir +180 ; l'essai Breakout interrompu a consommé 132, pas 72.
La géométrie des pixels/RAM n'est pas modifiée par cette correction comptable.
Les coûts historiques ne doivent plus être repris sans cet erratum. Le moteur
doit être corrigé avant les prochains essais ; aucun gate ni apprentissage
ARC 4 n'a été exécuté. Les anciennes captures scellées doivent être reproduites
dans leur commit d'origine, avec cet erratum, jamais avec un mélange de versions.
