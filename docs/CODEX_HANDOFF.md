# Ragnarok — point de reprise

10 septembre 2026, vers 00:36 Paris. Continuer d'un jalon au suivant sans final
aux commits. Arrêt seulement sur demande, travail achevé ou avis indispensable.
Jérémie a passé Astra en Maximum, déjà acquitté ; aucun réglage modifié ici.
L'audit partage d'écran/RC est annulé. Français simple, commentaires utiles réguliers.
Objectif : apprendre puis réutiliser pour apprendre plus vite. ARC2 : 3/6 vs 0/6,
delta 0,069 < bruit 0,076 ; accélération NON démontrée. Aucun entraînement Atari,
checkpoint Atari ou gate 0 ARC4. 768 mises à jour SYNTHÉTIQUES diagnostiques terminées.

Détails antérieurs : docs/history/CODEX_HANDOFF_20260910_0014.md, puis
CODEX_HANDOFF_20260909_2357.md,2338,2314,2232. Lire seulement les sections utiles.
Ne pas charger toute la session Claude, les logs entiers ou les configurations.

## ARC3 v62 — confirmation GPU active, cinq mondes terminés sur neuf

Contrôleur 20936 / enfant 22896 vérifiés à 00:26 : C:/Python314/python.exe -u -m
scripts.recover_confirm_v62 ; enfant -m scripts.confirm_v61 --lineage 1
--n-units 3 --b-max 6 --num-envs 64 --store-persists --g-units 2 --resume.
Lignée 1 démarrée 9 sept 18:41:46 Paris. s0 complet : 3 mondes / 25433 s,
15 JSON committés 9deb244, agrégat SHA
3d30c06fe7b6d5be84f042ae26e70c42f0ba70f09610df7eab43cbd7fe66b4ce.
s1_8310 terminé 6447 s + L ; s1_8311 terminé 13634 s + L ; s1_8312 M4 terminé
16760 s et Fa terminé 20698 s. À 00:35, Fb puis L restent en cours/à faire.
Aucun score/verdict intermédiaire ; ne pas committer les fichiers actifs.
Autorité ARC3_PLAN.md §12.6, gel a189ea5bb0f9c8b7e9016a50aefc1cbca603426f.
17 sources CRITICAL + 3 runners / états / checkpoints immuables (histoire).
Reprise 9 sept 11:37 après WindowsUpdate ; 2166 s interruption facturée,
budget de reprise 84742,0902837 s. Projection 34,3461 h / 42 sans extension.
Après confirm_complete_review_required : revue 9 mondes / budget / sources,
puis score gelé et figure 423b770 / af94a62. Suivi horaire existant
suivi-ragnarok-arc-3 ; pas de duplication, aucun autre entraînement GPU.

## ARC4 — fondations établies, limites conservées

Les six instruments et références sont revus. RAM uniquement dans le juge.
Boxing/Frostbite/Riverraid/Kangaroo jamais ouverts. Débits MsPacman sous les
1000/2000 proposés : original natif 580,7928/s, cache 539,3213/s, gather 523,8992/s.
Aucun gain de chaîne démontré ; anciennes sources, DLL et tentatives immuables.
Diagnostic des archives réelles : 72 pas / 69 P / 1300 ancres I / 6 paires,
0 optimizer / 0 émulation nouvelle / 0 GPU, fini. C : 150 instances propres,
ZERO supervision / perte / gradient sur les six paires. Revue ed4cc04.
Norme max avant clip réellement observée 120,42449 ; l'extrême 1e22 est SYNTHÉTIQUE.

Transport RGB entre Python : 9 cas / 1266 pas / 6 paires / 7454 charges,
images, empreintes complètes et comptes EXACTS de l'ancienne baseline ; aucun
modèle. Résultats ca94418, revue 09cb979 (SHA e93d2932c0ad5973558f51729dc189c99523ec5efbc08fce11fd7a93c06ca287).
Tous workers fermés. Pas de mesure de débit de cette variante ni de test à 16 natifs.
Parser ordonné 4e42567 : 448965 paramètres, paire RGB ordonnée seulement dans d,
anciens f/m/c/s poolés inchangés, effet seulement dans d. Revue 258fae5.
Flux ordonné 258fae5 + propriétaires fd0c704, revue 0b474ab : P4 / I exact8,
mémoire deux écrans propres, contrats avant leurs commandes payantes, coûts et
observations conservés sur échec. Pas d'inférence de cette variante sur Atari.
Anciennes fondations / sources / empreintes détaillées dans histoire 0014/2357.

ME / table de boutons / reader / drives / S / policy / ledger / fusion des
catégories restent absents. UNKNOWN nul ne conditionne pas le parser aux boutons.
P/C/ME ont une dépendance de démarrage encore ouverte. Lag1 n'est pas une preuve
universelle de contrôle direct ; lag2 n'est pas une preuve de contrôle indirect.
Gate 0 NON gelé : entrée / amorçage / coefficients / calendrier / registres et
coûts à résoudre avant GPU ; budgets et critères proposés restent conservés.

## Deux ajustements avec réseau — terminés, NÉGATIFS

Même paire de scènes 16×16, apprentissage et mesure sur ces scènes : pas de
généralisation mesurée. Seuil natif strict : L1 < 0,01171875. Source poolée
commune et futurs opposés distants de 0,046875 ; borne commune 0,0234375.

Original : protocole fb1aea3, préflight 95250f9 AVANT l'essai, 55 sources.
512 updates CPU (256 pooled + 256 ordered), 10 sept 00:02:50→00:03:46,
session 19056 FERMÉE. Résultats baedfb9 ; revue complète 24bc47c :
craft_v6_out/v70_synthetic_motion_fit/fit_review_v0.json
SHA 9d71b8ebe3bcdd6bd35e5cc79b5962ccb9293de252e436f125650f0c8c71ccbf.
Pooled final 0,0252121985 ; ordered final 0,025491314 ; critère NON atteint.
55 sources / 16 preuves vérifiées. Aucune réinférence dans la revue.

Perte à trois échelles (1/3/7) : helper 0611c84, revue 37a4d9e.
Protocole b4e01c2 ; préflight b0f7fea AVANT l'essai, 62 sources / 18 preuves :
026927cbe5eafecc0a9ff501d8a9851c83bbcbd533ce74e5acf7956beb1267fe.
256 updates CPU, mêmes poids vierges / mesures natives initiales exactes,
00:26:37→00:27:24 Paris, session 79982 FERMÉE. Résultats 961db60 :
craft_v6_out/v70_blurred_motion_fit/fit_v0.json
SHA 598226a743bfb404e5068ff74648c9564036b4f5e4be28f8644a2b20beef7bfc ;
attempt 0a5aba4286603bee59a1b10ff67bda7824f600b952879a347a203908dc6d368c ;
console 7ddde7c5055e7682678d2a24ec200d797071cbdfae528a89401465a5524e279d (-text).
Final natif 0,02534569799900055 > seuil ; séparation 0,0005912389024160802.
Exécution complète, résultat scientifique NÉGATIF ; 0 émulation / GPU / checkpoint.
review_recovery_v62 ACTIF pour fit_review_v0.json, sans réinférence ni optimizer.
COMMITTER son reçu avant le préflight du témoin ci-dessous.

## Prochain jalon immédiat — témoin des champs libres, PAS encore exécuté

3fea275 : scripts/check_free_displacement_fit_v70.py, tests du même nom,
docs/ARC4_FREE_DISPLACEMENT_FIT_{DRAFT.md,PROTOCOL.json} et revue scientifique
ARC4_MOTION_BOOTSTRAP_REVIEW_20260910.md (seconde revue, pas un protocole).
4 tests PASS 7,12 s, session 22204 FERMÉE, aucune vraie mise à jour.
- Deux champs indépendants nuls de forme 2×2×16×16 ; 1024 paramètres float32,
  sans réseau, RNG, direction, masque ou cible de déplacement.
- Même source / futur / splatting normalisé / perte L1 1/3/7. 256 updates Adam,
  lr0,05 en unités pixels, β.9/.999, eps1e-8, maxnorm1, weight_decay0.
- Mêmes mesures 0/1/8/32/128/256, seuil NATIF inchangé 0,01171875. Tableaux
  champs / RGB / couvertures finaux conservés pour diagnostic numérique.
- Un essai, CPU 300 s coopératives hors preuves, aucun retry / prolongation.
  Step marqué en mémoire avant appel, finally persisté en échec ; arrêt dur
  incertain, pas un journal disque à chaque mise à jour.
- Succès = faisabilité de l'ajustement libre dans ce protocole seulement ;
  aucun résultat ne prouve perception, généralisation, Atari ou transfert.
- proofs() attend la revue du résultat flouté. Sources attendues 67, fermeture
  réelle à vérifier indépendamment. Ne PAS lancer run() avant préflight scellé.
- Reçu futur : craft_v6_out/v70_free_displacement_fit/preflight_v0.json,
  kind free_displacement_fit_independent_preflight. Reviewer déjà informé.
- Après commit du préflight : une exécution CPU, préserver console/attempt/result,
  revue sans optimizer, puis continuer au jalon suivant.

## Environnement et coordination

Master, commits ciblés, pas de push / suppression. Exclure .codex/, ARC3 actif
(v61_confirm.json, s1/s2, états, checkpoints), anciens smokes/notch. Ne pas lire
.codex/config.toml. Console ignorée : add -f exact + .gitattributes locale -text.
C:/Python314/python.exe : Python3.14.3, Torch2.11.0+cu126, numpy2.4.4,
scipy1.17.1, pas ale_py. Tests CPU : CUDA_VISIBLE_DEVICES vide, OMP/OPENBLAS/MKL=1,
torch threads1. .venv-arc4/Scripts/python.exe : ALE.12.1, numpy2.4.4,
scipy1.18.1, pas Torch. Pas de réinstallation ni rebuild pendant ARC3.
Ryzen53600X6c/12t, RTX4080 16Go. Toutes sessions racine CPU fermées à00:36.
Revue indépendante explicitement prévue par AGENTS : réutiliser
review_recovery_v62 ; audit_motion_bootstrap terminé, document committé3fea275.
Ancien audit_gate_v62 annulé/pending_init : ne pas réveiller. Ne pas finir sur
un commit ou une attente : poursuivre le travail autorisé.
