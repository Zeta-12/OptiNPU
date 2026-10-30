# OptiNPU — Plan du projet

## Contexte matériel

- CPU : Intel Core Ultra 5 235U
- NPU : Intel AI Boost (~11-13 TOPS INT8)
- iGPU : Intel Graphics
- RAM : 15,5 Go
- Python : environnement virtuel 3.12 (3.14 trop récent pour OpenVINO / PyTorch XPU / NNCF)

## Contraintes structurantes

1. **Le NPU Intel ne fait que de l'inférence** (OpenVINO, plugin NPU). L'entraînement se fait sur iGPU (PyTorch XPU) ou CPU, puis le modèle est exporté et exécuté sur le NPU. L'entraînement sur NPU reste une piste exploratoire (L5).
2. **Pas de record d'AP COCO.** L'objectif retenu (révisé le 2026-10-27) : **égaler l'AP de l'état de l'art avec un modèle beaucoup plus léger au sens du NPU**, exécuté entièrement sur le NPU (voir Enjeu 2). Les entraînements COCO complets nécessiteront du GPU cloud.
3. **Licence Apache 2.0**, sans dépendance à Ultralytics (AGPL).

---

## Enjeu 1 — Librairie `optinpu`

### Expérience utilisateur cible

```python
from optinpu import Detector
model = Detector("optidet-s", pretrained=True)
model.train(data="mon_dataset.yaml", epochs=50)   # iGPU/CPU, automatique
model.export(target="npu", precision="int8")      # PTQ ou QAT, calibration automatique
model.predict("image.jpg", device="npu")          # OpenVINO NPU
model.benchmark(device=["cpu", "gpu", "npu"])     # latence, débit
```

### Architecture

| Module | Rôle |
|---|---|
| `data/` | Lecture COCO / YOLO / VOC, augmentations (mosaic, mixup, copy-paste), dataloader |
| `models/` | Registre de modèles, backbone / neck / tête interchangeables, adaptateurs vers RT-DETR, D-FINE, RF-DETR (baselines) |
| `engine/` | Boucle d'entraînement (AMP, EMA, accumulation, callbacks), sélection automatique du device (`xpu` puis `cpu`) |
| `export/` | PyTorch → OpenVINO IR, formes statiques, NNCF (INT8 PTQ puis QAT) |
| `runtime/` | Inférence OpenVINO NPU : cache de compilation, requêtes asynchrones, prétraitement dans le graphe |
| `npu_lint` | Détecte les opérations qui retombent sur CPU ou sont lentes sur NPU (grid_sample, formes dynamiques, NMS…) |
| `bench/` | mAP (pycocotools / faster-coco-eval), profilage par couche, courbes précision-latence |
| CLI | `optinpu train | export | predict | bench` |

### Jalons

| Phase | Contenu | Critère de sortie |
|---|---|---|
| L0 | Environnement Python 3.12, OpenVINO, PyTorch XPU, NNCF ; test « hello NPU » | Un ResNet tourne sur le NPU, latence mesurée |
| L1 | Inférence : détecteur pré-entraîné exporté et exécuté sur NPU | `predict(device="npu")` fonctionne, sorties cohérentes avec le CPU |
| L2 | Quantification INT8 (PTQ NNCF) et module `bench` | Écart mAP INT8 vs FP16 connu, tableau de latences CPU/GPU/NPU |
| L3 | Entraînement : fine-tuning sur petit dataset sur iGPU | Pipeline complet entraînement → export → NPU |
| L4 | `npu_lint`, QAT, CLI, docs, tests, packaging pip | v0.1 publiable |
| L5 | Exploration : entraînement partiel sur NPU (tête, LoRA) | Note de faisabilité |

---

## Enjeu 2 — Modèle « OptiDet »

### Objectif (révisé le 2026-10-27)

**Égaler l'AP COCO des détecteurs temps réel de l'état de l'art avec un modèle beaucoup plus léger au sens du NPU** : exécuté entièrement sur le NPU (ni iGPU ni CPU), INT8 autorisé, exact selon `npu_lint`, et **4 à 5 fois plus rapide** que le meilleur placement actuel du modèle de référence.

- **« Léger » se mesure en latence NPU, pas en paramètres ni en FLOPs.** D-FINE-N (3,8 M de paramètres) prend 43 ms sur le NPU en mode exact, RF-DETR-N 106 ms, alors que ResNet-50 tourne en 4 ms à 224 px. Le NPU est limité par le nombre et l'irrégularité des ops, pas par le calcul.
- **NPU seul**, parce que c'est ce qui donne sa valeur au NPU : CPU et iGPU restent libres, et la consommation est faible. Le mode hybride `npu+gpu` sert de point de comparaison, pas de cible.
- **Pas de record d'AP** : il faudrait de gros modèles et des centaines d'heures de GPU, qui tourneraient mal sur un NPU.

Paliers (cibles issues de M0, à confirmer après les micro-benchmarks de M1) :

| Palier | AP COCO val2017 visée | Références (AP mesurée sur ce banc) | Référence la plus rapide | Latence NPU seul visée |
|---|---|---|---|---|
| N | ≈ 43 | DEIMv2-N 42,9 · D-FINE-N 42,8 · YOLO26-N 40,3 | DEIMv2-N, iGPU 15,1 ms | 3,0 à 3,8 ms |
| S | ≈ 48,5 | D-FINE-S 48,5 · RF-DETR-N 48,4 · RT-DETRv2-R18 48,1 · YOLO26-S 47,7 | RF-DETR-N, iGPU 23,0 ms | 4,6 à 5,8 ms |

**Protocole de mesure** : COCO val2017 complet (5 000 images), batch 1, latence du modèle seul (médiane), placement NPU seul exact, FP16 ou INT8, machine sur secteur. Banc : `benchmarks/m0/`.

### État de l'art (vérifié le 2026-10-27)

| Famille | Points clés | Licence | Statut pour OptiDet |
|---|---|---|---|
| RF-DETR (ICLR 2026) | Backbone DINOv2 fenêtré, NAS ; N 48,4 AP à 384 px, S 53,0 à 512 px | Apache 2.0 (N à L) | Référence ; enseignant possible |
| DEIMv2 | Backbone distillé de DINOv3 (S, M) ou HGNetv2 (N et moins) ; N 43,0, S 50,9 | Non commerciale | Mesuré uniquement, aucun code ni poids réutilisé |
| D-FINE | Raffinement des boîtes par distributions ; N 42,8, S 48,5 | Apache 2.0 | Référence ; enseignant possible (D-FINE-L/X) |
| YOLO26 | Pur CNN, tête un-à-un sans NMS ; N 40,1, S 47,8 (bout en bout) | AGPL-3.0 | Mesuré uniquement ; concurrent le plus proche sur NPU |
| RT-DETRv2 | R18 47,9 | Apache 2.0 | Référence |

Sources : [dépôt RF-DETR](https://github.com/roboflow/rf-detr), [dépôt DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) et [sa licence](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md), [article DEIMv2](https://arxiv.org/pdf/2509.20787), [analyse YOLO26](https://arxiv.org/pdf/2601.12882).

### Thèse : un détecteur conçu pour le NPU (mise à jour avec L1-L5)

1. **Tout le graphe sur le NPU** : pas de NMS (appariement un-à-un), sélection de requêtes simple.
2. **Peu d'ops, larges et régulières** : canaux multiples de 32 ou 64, pas de couches étroites à haute résolution. C'est la condition pour que l'INT8 accélère vraiment (×2,4 sur ResNet-50, aucun gain sur D-FINE).
3. **Pas d'attention déformable** (`GridSample` lent), et toute attention validée par `npu_lint`. Le compilateur répartit mal l'attention sur 2 tuiles (jusqu'à −3,6 AP).
4. **Conçu pour l'INT8** : activations bornées, convolutions reparamétrables, normalisations fusionnables, QAT.
5. **Tête robuste au FP16** : le raffinement par distributions de D-FINE encaisse le bruit du NPU (erreur 0,025 contre 0,17 pour une régression directe).
6. **Distillation** depuis un gros enseignant sous licence compatible (D-FINE-L/X, RF-DETR-L, features DINOv2 ; licence DINOv3 à vérifier) : c'est le levier principal pour atteindre l'AP de l'état de l'art avec un petit modèle.
7. **Résolution** comme levier (RF-DETR-N atteint 48,4 AP à 384 px).
8. **Recherche d'architecture guidée par la latence NPU mesurée** (`optinpu.bench` et `npu_lint` dans la boucle), pas par les FLOPs.

### Jalons

| Phase | Contenu | Critère de sortie |
|---|---|---|
| M0 | Référence : baselines N et S mesurées sur ce matériel (AP val2017 complet ; latences CPU, iGPU, NPU 1 et 2 tuiles, hybride ; FP16 et INT8 ; lint) | Frontière de Pareto « NPU seul » ; cibles de latence des paliers fixées |
| M1 | Briques NPU : micro-benchmarks de blocs (convolutions, attention, têtes, sélection de requêtes) en FP16 et INT8 sur NPU | Catalogue de blocs et règles de conception chiffrées ; premier squelette d'OptiDet |
| M2 | Prototype OptiDet : distillation, ablations sur un sous-ensemble de COCO (iGPU), QAT INT8, recherche guidée par la latence NPU | Prototype dans la zone visée de latence, tendance d'AP crédible |
| M3 | Entraînement COCO complet (GPU cloud) | Paliers N et S atteints sur val2017 complet |
| M4 | Publication : poids, intégration `Detector("optidet-n/s")`, rapport technique | v1.0 |

---

## Enjeu 3 — OptiNPU, profileur et optimiseur pour le NPU (décidé le 2026-10-29)

### Objectif

Faire de `optinpu` l'outil qu'on utilise quand on veut déployer **son propre modèle** sur un NPU : il dit où part la latence, si le résultat est juste, et quels changements la réduisent sans perte d'AP (ou presque), mesures à l'appui. M3 (OptiDet sur COCO complet) passe après.

Décisions de l'utilisateur :
- **Entrée** : tout modèle (IR OpenVINO, ONNX, PyTorch). La garde d'exactitude est enfichable : AP et accord des détections pour un détecteur, accord des sorties sinon. Les détecteurs sont le cas de référence.
- **Autonomie** : trois niveaux de réécritures (exactes, approchées, avec réentraînement court). Les deux premiers s'appliquent automatiquement sous un budget d'AP ; le troisième est proposé avec son gain mesuré, puis récupéré automatiquement si l'utilisateur fournit des images.
- **Article** : l'outil est la contribution centrale ; la caractérisation du NPU et les dégradations silencieuses en sont la motivation.
- **Matériel** : ce seul NPU (3720). Limite à assumer dans l'article ; la lib reste centrée sur OpenVINO.
- **Cible : le réseau entier sur le NPU** (décidé le 2026-10-29, après le paquet 3 de P3.4). Seule la sélection finale légère (TopK / NMS et ce qui suit, sans conv ni matmul) peut aller sur le CPU, comme tout post-traitement côté hôte. Le placement d'une partie lourde (décodeur DETR, fin de l'encodeur) sur l'iGPU ou le CPU n'est qu'une option (`optimize(..., offload=("gpu",))`, `--offload gpu`) qui sert de comparaison en P5. Raisons : le NPU vaut par le CPU et l'iGPU qu'il laisse libres, à faible consommation ; et sur les DETR, l'outil ne faisait que retrouver le placement hybride de M0 (D-FINE-N 15,4 contre 15,5 ms ; RF-DETR-N hybride 37,4 ms, plus lent que l'iGPU seul à 23,0 ms).

### Idée directrice

**Sur un NPU, la latence ne dépend pas des valeurs des poids.** On peut donc mesurer exactement, avec des poids quelconques, la latence de n'importe quelle variante d'un modèle, avant de payer l'entraînement ou même une calibration. Les approches existantes prédisent la latence (nn-Meter : 83 % de précision seulement sur le VPU Intel) ou optimisent sans vérifier ce que fait le compilateur, alors que celui-ci peut faire perdre jusqu'à 4,6 AP sans erreur (M0, 2 tuiles). OptiNPU mesure les deux : le gain réel de latence sur l'appareil et l'effet sur l'AP.

### Architecture

| Pilier | Rôle | Acquis réutilisés |
|---|---|---|
| **Profileur** | Où part le temps, mesuré : latence par profondeur (graphes tronqués), latence économisée en contournant chaque région (bloc, type d'op), « what-if » d'une réécriture ; compteurs du constructeur en complément (ils se chevauchent sur le NPU) | `profile`, `benchmark_stages`, suite S7 de M1 (coût marginal) |
| **Validateur** | Est-ce juste sur le NPU : accord des détections (ou des sorties), bisection vers l'op fautive, compilations dans un sous-processus surveillé (mémoire, délai, perte du périphérique) | `lint`, `diagnose`, `benchmarks/watchdog.py` |
| **Optimiseur** | Réécritures candidates, latence mesurée, impact sur l'AP mesuré, boucle gloutonne sous budget, rapport ; modèle optimisé en sortie | `npu_patches`, `quantize`, QAT, distillation |

Abstractions :
- **Tâche** : prétraitement, décodage des sorties et métrique (détection : formats `optinpu`, `detr`, `dets`, redimensionnement étiré ou letterbox, comme le banc M0 ; générique : similarité des sorties).
- **Réécriture** : motif cherché dans l'IR, transformation, classe (`exact`, `approx`, `retrain`). Une réécriture `retrain` appliquée à l'IR donne la latence exacte du modèle réentraîné, mais pas son AP.

Catalogue initial (d'après M0-M2) :

| Classe | Réécritures |
|---|---|
| Exactes | TopK / NMS / post-traitement hors du graphe (1,2 à 2,5 ms en M1) ; depthwise en conv dense équivalente (débloque l'INT8) ; canaux complétés au multiple de 16 ; pixel shuffle en conv transposée ; formulations équivalentes d'une activation ; sorties FP16 ; gel des constantes et patches existants ; nombre de tuiles, y compris par partie (décodeur sur 1 tuile, reste sur 2), sous contrôle d'exactitude |
| Approchées, sans réentraînement | INT8 en précision mixte guidée par la sensibilité de chaque couche **et** son coût de latence mesuré (généralise l'analyse PTQ de M2.5) ; résolution ; moins de requêtes ou de couches de décodeur (DETR) ; approximations d'activations (GELU tanh) |
| Avec réentraînement court | SiLU / HardSwish / GELU → ReLU ; LayerNorm → BatchNorm ; bilinéaire → plus proche voisin ; attention → mélangeur convolutif ; reparamétrisation RepVGG. Récupération : distillation depuis le modèle d'origine (sans étiquettes), puis QAT |

Protocole : la recherche (choix des réécritures) se fait sur des images de train2017 ; l'AP rapportée est celle de val2017 complet. Aucune décision n'est prise sur les images du rapport.

### Jalons

| Phase | Contenu | Critère de sortie |
|---|---|---|
| P0 | Socle : tâches (détection, générique), chargement IR / ONNX / PyTorch, compilation surveillée en sous-processus ; banc d'évaluation (les 9 modèles de M0, OptiDet-N, CNN de M1) | Les 10 détecteurs passent par l'API générique avec les AP de M0 |
| P1 | Profileur mesuré : latence par profondeur, contournement de régions, what-if ; regroupement par module d'origine | Ses prédictions de gain collent mieux aux gains réels que les compteurs du constructeur et qu'un modèle de coût additif |
| P2 | Cadre de réécritures sur l'IR, garde d'exactitude, réécritures exactes | Gain mesuré sur les 10 modèles, ΔAP nul |
| P3 | Réécritures approchées, précision mixte guidée par la latence, boucle sous budget, rapport, CLI `optinpu optimize`. Découpé en étapes courtes, avec une pause après chacune (le PC sert aussi à autre chose) : P3.1 INT8 comme candidat approché et budget d'AP (~30 min de NPU) ; P3.2 précision mixte guidée par la latence (~45 min) ; P3.3 rapport et CLI ; P3.4 optimiseur sur les détecteurs, par paquets de 2-3 modèles, avec reprise (paquets 1 à 3 avec décodeur sur iGPU autorisé, gardés comme comparaison) ; P3.5 même banc en NPU seul : 7 détecteurs et 3 classifieurs ImageNet, en 3 paquets | Accélération à ΔAP ≤ 0,3-0,5 sur les 10 modèles |
| P4 | Réécritures avec réentraînement : what-if côté PyTorch, récupération par distillation sans étiquettes et QAT | Démontré sur D-FINE-N et un CNN (licences compatibles) |
| P5 | Évaluation pour l'article : ablations, comparaisons (défauts OpenVINO, PTQ par défaut, guidage par les compteurs du constructeur, placement hybride) | Tableaux et figures reproductibles |

### Article (contribution centrale : l'outil)

- **Motivation** : caractérisation d'un NPU client Intel (M0, M1) et dégradations silencieuses d'AP dues au compilateur (2 tuiles, YOLO26-S qui fait tomber le NPU).
- **Contributions visées** : (1) profilage mesuré, valable malgré les fusions et le chevauchement des tâches ; (2) garde d'exactitude fondée sur la tâche, calibrée (M2.1) ; (3) optimiseur à trois niveaux avec what-if indépendant des poids ; (4) évaluation sur 10 détecteurs.
- **Travaux à situer** : nn-Meter, NetAdapt, HAQ, MobileDets, EfficientNet-EdgeTPU, HOT (Samsung), Grassroots Operator Search (IBM, DATE 2025), optimiseur de graphe d'Arm, TASO / PET, tests de compilateurs DL (NNSmith).
- **Limite assumée** : un seul NPU (3720, pilote 32.0.100.4778).
- **Cible** : à décider (prépublication arXiv, puis atelier ou revue).
- **V0** (2026-10-30) : `paper/main.tex` + `references.bib`, template IEEEtran (conférence), en anglais, environ 5 400 mots (au-delà des 6 pages habituelles). Écrit pour vérifier la maturité de l'outil et des résultats ; tout ce qui manque est marqué `\todo{}` en rouge (31 marques), surtout les résultats DETR en NPU seul (P3.5), la récupération après réentraînement (P4), la campagne sur machine au repos et l'AP sur val2017 complet (P5). Les chiffres de validation du profileur sont recalculés sans OptiDet (5 modèles, 150 interventions, 73 significatives).

---

## Suivi

- [x] L0 — 2026-10-27. OpenVINO 2026.4.1, PyTorch 2.14.1+xpu, NNCF 3.4. ResNet-50 FP16 : CPU 31,5 ms, GPU 4,3 ms, NPU 4,1 ms (287 fps).
- [x] L1 — 2026-10-27. `Detector`, export, runtime, CLI, `diagnose`, 19 tests. D-FINE-N/S correctes sur NPU ; RT-DETRv2 dégradé sur NPU (correct sur GPU FP16).
- [x] L2 — 2026-10-27. mAP COCO (sous-ensemble val2017), INT8 NNCF, profilage par couche et par étage, CLI `export | eval | profile`, 22 tests. Conclusion : l'INT8 n'accélère pas D-FINE sur NPU (voir journal).
- [x] L3 — 2026-10-27. `Detector.train()` / `optinpu train` sur dataset au format COCO, iGPU (XPU), EMA, tête réinitialisée si nouvelles classes, `from_checkpoint`. D-FINE-N fine-tuné sur COCO-animaux (10 classes) : 62,6 AP (PyTorch), 63,0 AP sur NPU, 26 ms. 26 tests.
- [x] L4 — 2026-10-27. `npu_lint` (CLI `lint`), `diagnose.locate_divergence_split`, exécution hybride `npu+gpu` (la plus rapide sur les 3 modèles), RT-DETRv2 exact sur NPU (`NPU_TILES=1`), packaging v0.1.0 (wheel, licence Apache-2.0), dépôt git local, 32 tests. QAT reportée à M2 : sans intérêt tant que le modèle ne profite pas de l'INT8 sur ce NPU.
- [x] L5 — 2026-10-28. Note de faisabilité ci-dessous. Livrable : `train(..., backbone_device="npu")` (backbone gelé sur NPU, ×2 sur le fine-tuning). Calcul de gradients sur NPU abandonné. Une seule tuile NPU devient le réglage par défaut des modèles DETR. Nettoyage final de la librairie (ruff, code mort retiré), 35 tests.

- [ ] P1 / P2 — en cours (2026-10-29). Profileur mesuré (`optinpu.profile_measured`, `optinpu profile --measured`) ; test de la prémisse (P1.0 : poids denses sans effet, parcimonie exploitée, ops dépendantes des données) ; cadre de réécritures (`optinpu.rewrites`) et optimiseur (`optinpu.optimize`, `optinpu optimize`) ; prédictions et mesures toutes en sous-processus. YOLO26-N 3,0× et YOLO26-S 2,1× plus rapides, exacts, défauts 2 tuiles corrigés. Validation P1 (P1.2, P1.3) avec répétitions adaptatives et intervalles de confiance : sur les interventions significatives des 6 modèles, la mesure répétée (corrélation +0,68, erreur 1,0 %, regret 0) bat les compteurs du constructeur (+0,56, 3,4 %, regret 0,26) et le modèle additif (+0,45, 11,1 %, regret 0,40) ; sur les modèles de moins de 8 ms, l'ordre fin des gains reste au niveau du bruit. 81 tests. À faire : campagne sur machine au repos (article), optimiseur sur les 10 détecteurs et les CNN, INT8 et précision mixte (P3), rapport.
- [x] P0 — 2026-10-29. Socle générique : `optinpu.tasks` (`DetectionTask` aux formats `optinpu` / `detr` / `dets`, étiré ou letterbox ; `OutputTask` ; `Deployed`), `export.load_model` (IR, ONNX, PyTorch), `optinpu.sandbox` (jobs en sous-processus surveillé : latence, sorties, comparaison round-robin, compteurs constructeur). Avant de rebâtir le banc M0 dessus, la lib donnait des détections identiques au banc M0 sur les 9 baselines, et OptiDet-N retrouvait son AP de M2.5 (9,7 CPU, 10,0 NPU). Banc P0 : 10 détecteurs + 7 CNN torchvision (`benchmarks/p0/`), tous exécutés via l'API générique. `import optinpu` ne charge plus PyTorch (3,7 s au lieu de 8 s par job isolé).
- [x] M2 — 2026-10-28. `npu_lint` fondé sur les détections, architecture OptiDet et perte d'entraînement (`optidet-n`, backbone RepVGG-A0 ImageNet), balayage de latence de 36 candidats, premier entraînement (10 000 images COCO, 10 epochs : 8,5 AP), distillation depuis D-FINE-S (10,0 AP, +18 %), QAT (INT8 sur NPU 10,0 AP, aucune perte, 3,37 ms, exact en 2 tuiles) intégrée à la librairie (`Detector.train_qat`, CLI `optinpu qat`), 55 tests. L'AP atteignable reste à mesurer en M3 (COCO complet).
- [x] M1 — 2026-10-28. 396 blocs (7 suites) mesurés en FP16/INT8 × 1/2 tuiles, rapport dans `benchmarks/m1/RESULTS.md`. Règles de conception et squelette d'OptiDet dans le journal (section M1). Reporté en M2 : critère d'exactitude de `npu_lint` fondé sur les détections.
- [x] M0 — 2026-10-27. 9 baselines × (CPU, iGPU, NPU 1 et 2 tuiles, hybride) × (FP16, INT8) sur val2017 complet : 181 mesures, rapport dans `benchmarks/m0/RESULTS.md`. Conclusions dans le journal (section M0).

## Journal technique NPU

Pièges du compilateur NPU rencontrés (driver 32.0.100.4778, OpenVINO 2026.4) et réponses dans la librairie :

| Problème | Symptôme | Réponse |
|---|---|---|
| MaxPool k2 s1 après Pad partagé avec une conv (stem HGNetv2) | Valeurs fausses dès le stem, 0 détection, 134 ms | `MaxPool2x2s1` (max de 4 tranches) |
| Pooling `ceil_mode=True` (ResNet-vd) | Compilation échoue (« Bad optional access ») | `FloorWhenExactPool` |
| Ancres invalides = `finfo(float32).max` | Inf en FP16 | Gel des ancres en FP32, bornées à 1e4 |
| Embeddings sin-cos recalculés (`10000 ** x`) en FP16 | Erreur relative de 8,7e3 | Gel en buffers constants |
| Self-attention du décodeur répartie sur 2 tuiles NPU (RT-DETRv2 ; D-FINE légèrement touché) | Scores dégradés (erreur 0,42), chaque op isolée juste ; sorties identiques au bit près quelle que soit la réécriture du graphe | `NPU_TILES=1` (exact, mais 81 ms au lieu de 47), ou exécution hybride `npu+gpu` |
| `EXECUTION_MODE_HINT=ACCURACY` sur NPU | 0 détection (D-FINE et RT-DETRv2) | Ne pas utiliser |
| Graphes de rétropropagation de convolutions | Compilateur à 53-55 Go de mémoire virtuelle, machine saturée | Ne pas compiler de gradients de convolutions sur NPU (piste L5 abandonnée) |
| Mode 2 tuiles, hors DETR aussi (M0) | YOLO26-N −4,6 AP, RT-DETRv2 −3,8, DEIMv2-N −0,9 ; D-FINE et RF-DETR intacts | 1 tuile, ou valider chaque modèle en 2 tuiles par l'AP |
| YOLO26-S en 2 tuiles sur images réelles (M0) | Perte du périphérique NPU (`ZE_RESULT_ERROR_DEVICE_LOST`) après 4 à 7 inférences sur des images différentes ; une entrée répétée passe ; FP32 et FP16 touchés ; le NPU se rétablit seul | `NPU_TILES=1`. Rapport et reproducteur pour Intel : `reports/intel-npu-device-lost/` |
| Critère d'exactitude de `npu_lint` (M0) | Écart maximal des sorties mal corrélé à l'AP : RF-DETR signalé à 0,06 mais exact ; YOLO26-N à 0,05 (limite) mais −4,6 AP | À remplacer par une comparaison des détections (M1) |

Mesures D-FINE (640×640, FP16, batch 1) :

| Modèle | CPU | GPU | NPU |
|---|---|---|---|
| D-FINE-N | 38 ms | 17 ms | 26 ms |
| D-FINE-S | 119 ms | 43 ms | 47 ms |

### L2 — précision et INT8 (500 premières images de val2017, calibration sur 300 autres)

| Modèle | Device | Précision | AP | AP50 | Latence |
|---|---|---|---|---|---|
| D-FINE-N | CPU | FP16 | 45,8 | 62,8 | 35 ms |
| D-FINE-N | NPU | FP16 | 45,6 | 62,2 | 26 ms |
| D-FINE-N | NPU | INT8 | 44,7 | 61,9 | 36 ms |
| D-FINE-N | CPU | INT8 | 44,7 | 61,9 | 19 ms |
| D-FINE-S | NPU | FP16 | 51,2 | 67,7 | 47 ms |
| D-FINE-S | NPU | INT8 | 50,4 | 66,9 | 54 ms |
| D-FINE-S | CPU | INT8 | 50,4 | 66,9 | 49 ms |

(Les AP sont plus élevés que les valeurs publiées sur val2017 complet, 42,8 et 48,5 : effet du sous-ensemble. Seules les comparaisons internes comptent.)

Latence réelle par étage, D-FINE-N sur NPU (sous-graphes tronqués) :

| Étage | FP16 | INT8 (NNCF transformer) |
|---|---|---|
| Backbone HGNetv2-B0 | 10,1 ms | 20,1 ms |
| + encodeur hybride | 16,3 ms | 24,4 ms |
| Modèle complet | 25,8 ms | 35,5 ms |

Variantes INT8 de D-FINE-N testées : backbone seul (25,6 ms, AP 45,2), mode CNN partout (25,0 ms, AP 40,8), mode CNN sans décodeur (24,9 ms, AP 42,7). **Aucune n'accélère.**

Contrôle : ResNet-50 à 640×640 passe de 28,7 ms (FP16) à 11,8 ms (INT8) sur le même NPU, donc l'INT8 fonctionne. Les LAB d'HGNetv2 ne sont pas en cause (convnet avec LAB : ×1,43).

### L3 — entraînement sur iGPU

- **Débit** de D-FINE-N à 640×640 : CPU 1,5 img/s (batch 2) ; iGPU 3,7 img/s (batch 2, 0,6 Go) puis **5,1 img/s (batch 8)**.
- **Compilation JIT des kernels XPU** : environ 17 minutes au premier pas, en mémoire vive seulement par défaut. La librairie active `SYCL_CACHE_PERSISTENT=1` à l'import pour ne la payer qu'une fois.
- **Fine-tuning COCO-animaux** : 10 classes, 733 images d'entraînement et 99 de validation, prises dans val2017 [500, 4500) et disjointes du benchmark et de la calibration. Têtes réinitialisées, 12 epochs en batch 8, environ 30 minutes.

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 8 | 11 |
|---|---|---|---|---|---|---|---|---|
| AP val | 12,7 | 46,4 | 51,5 | 59,9 | 62,0 | 62,6 | 62,0 | 62,6 |

Plateau dès l'epoch 5 : avec si peu de données, 6 epochs suffisent.

- **Modèle fine-tuné exporté** : CPU 62,5 AP / 42 ms, **NPU 63,0 AP / 26 ms**. Les patches et le gel des constantes fonctionnent aussi sur un modèle réentraîné.
- Limites actuelles : augmentations minimales (retournement et couleur, sans mosaic ni crop), pas de reprise d'entraînement, et un dataset de démonstration tiré de COCO (le backbone connaît déjà ces animaux). À valider sur un vrai domaine nouveau.

### L4 — placement hybride et lint

La découpe NPU/CPU (`locate_divergence_split`) a localisé le bug de RT-DETRv2 sur la self-attention de la 1re couche du décodeur. Les réécritures de l'IR (SDPA décomposée, transposée) ne changent rien : le compilateur NPU normalise toutes ces variantes. En revanche, `NPU_TILES=1` est exact.

Mesures sur 500 images COCO val, 640×640, batch 1 :

| Modèle | CPU | iGPU | NPU | NPU (tête) + iGPU (décodeur) |
|---|---|---|---|---|
| RT-DETRv2-R18 | 50,8 AP · 227 ms | 50,9 · 44,0 | 50,8 · 81,0 (1 tuile) | **50,8 · 33,9** |
| D-FINE-S | 51,6 · 128 | 51,4 · 40,7 | 51,2 · 46,2 | **51,5 · 33,6** |
| D-FINE-N | 45,8 · 48 | 45,8 · 16,6 | 45,6 · 25,2 | **45,7 · 15,9** |

- Le NPU exécute le backbone et l'encodeur (convolutions larges), l'iGPU le décodeur (petites ops). Seuls 4 à 6 tenseurs transitent, sans copie, ce qui fait gagner 10 ms par rapport à un chaînage naïf.
- Le décodeur sur 1 tuile NPU est 2 fois plus lent que sur 2 tuiles : la répartition sur 2 tuiles accélère réellement, mais elle est fautive pour l'attention.

### L5 — entraîner avec le NPU : note de faisabilité

**Piste 1 — calculer des gradients sur le NPU (abandonnée, code retiré).** La passe avant et la rétropropagation étaient tracées en graphe statique (`torch.func.grad` + `make_fx`), converties en OpenVINO et compilées sur le NPU ; les poids étaient des entrées, les gradients des sorties, la mise à jour se faisait sur l'hôte.

- **Précision** : bonne avec le loss scaling (×1024), cosinus de 0,9997 ou plus avec le gradient FP32. Sans loss scaling, les gradients de convolution sont **tous nuls** (sous-dépassement FP16). Un classifieur entraîné 400 pas avec les gradients du NPU converge exactement comme avec ceux du CPU (loss 2,116 → 0,404, précision 91,9 % contre 91,8 %).
- **Vitesse**, pour un pas complet (passe avant + rétropropagation) :

| MLP (tokens × largeur) | CPU OpenVINO | iGPU OpenVINO | **NPU** | PyTorch iGPU |
|---|---|---|---|---|
| 300 × 256 | 4,9 ms | 11,4 | 7,6 | 1,7 |
| 2048 × 512 | 76 | 126 | **27** | 18 |
| 4096 × 1024 | 529 | 629 | **139** | 136 |
| 8192 × 1024 | 1083 | 1252 | **297** | 259 |

  Sur les convolutions (64 canaux, 40×40), le NPU met 59 ms contre 12 ms sur CPU.
- **Pièges du compilateur NPU propres à la rétropropagation** :
  - `ConvolutionBackpropData` avec peu de canaux donne un résultat faux ; contourné par une convolution ordinaire à noyau retourné quand le pas vaut 1.
  - Le gradient des poids calculé comme une convolution dont les canaux sont le batch est faux pour les petits batchs ; contourné par un MatMul par position du noyau.
  - Le découpage avec un pas de 2 ne compile pas (« Max striding levels exceeded »), et sa reformulation par reshape 6D plante parfois (« bad allocation »). **Les convolutions à pas supérieur à 1 ne sont pas fiables.**
  - Le graphe de gradients doit être replié (constant folding) avant compilation.
  - `cross_entropy` doit être écrite explicitement, sa décomposition ne se convertit pas.
  - **Incident** : sur certains graphes de gradients de convolutions, le compilateur NPU (qui tourne dans le processus Python) monte à 53-55 Go de mémoire virtuelle sans échouer. La mémoire de la machine sature (alertes « Resource-Exhaustion-Detector » dans le journal Windows), ce qui a fait planter Claude Code et Edge. **Ne jamais compiler sur le NPU un graphe non éprouvé sans surveiller la mémoire, et vérifier qu'aucun processus de compilation ne survit à une interruption.**
- **Verdict** : faisable pour des têtes MLP ou à base de transformers (MatMul), où le NPU égale l'iGPU. **Pas rentable pour un détecteur convolutif** : plus lent que le CPU, fragile, et l'iGPU est meilleur. Piste abandonnée le 2026-10-28 ; le code (`npu_grad`) a été retiré.

**Piste 2 — backbone gelé exécuté sur le NPU** (`train(..., backbone_device="npu")`). Le NPU calcule les features du batch suivant pendant que l'iGPU entraîne l'encodeur et le décodeur sur le batch courant.

| Fine-tuning D-FINE-N, COCO-animaux, batch 8 | Durée par epoch | Débit | AP val (CPU) | Epochs |
|---|---|---|---|---|
| Complet, iGPU (L3) | 142 s | 5,1 img/s | 62,6 | 12 |
| Backbone gelé, iGPU (A) | 98 s | 7,4 img/s | 62,5 | 8 |
| **Backbone gelé, NPU en parallèle (B)** | **70 s** | **10,4 img/s** | 62,1 | 8 |

- **×2 sur le temps d'entraînement** pour la même précision à 0,5 point près. Pas idéal mesuré sur l'iGPU : 1 514 ms complet, 1 007 ms backbone gelé, 666 ms avec features fournies. Le NPU produit 22,7 img/s de features, ce qui suffit.
- Le compilateur NPU journalise « MultiClusterStrategyAssignment … unsupported batch size » sur le backbone en batch 8 : bruit, la compilation réussit.

**Découverte en vérifiant les modèles fine-tunés** : le défaut de l'attention répartie sur 2 tuiles touche aussi D-FINE.

| Modèle fine-tuné (val animaux) | CPU | NPU 2 tuiles | NPU 1 tuile | NPU+iGPU |
|---|---|---|---|---|
| Complet | 62,5 | 63,0 | 62,4 | 62,3 |
| Backbone gelé (A) | 62,4 | **58,8** | 62,2 | 62,1 |
| Backbone gelé sur NPU (B) | 62,3 | 61,3 | 62,0 | 62,1 |

`npu_lint` signale les 3 modèles (écart de 0,18 à 0,61) et recommande une seule tuile. **`NPU_TILES=1` est désormais le défaut pour toute la famille DETR** : une perte silencieuse de 3,6 AP est pire qu'une latence plus élevée (D-FINE-N : 25 → 43 ms en NPU seul). Le mode hybride `npu+gpu` reste le plus rapide et il est exact (15,7 ms).

### M0 — référence sur ce matériel (COCO val2017 complet, batch 1, sur secteur)

Le banc reproduit les AP publiées à ±0,2 pour 8 modèles sur 9 (exception : DEIMv2-S, 46,9 contre 50,9, export ONNX du backbone DINOv3 suspecté).

| Modèle | AP | Meilleur placement (ms) | Meilleur NPU seul (ms, AP) | NPU seul / meilleur |
|---|---|---|---|---|
| YOLO26-N | 40,3 | iGPU 10,9 | 1 tuile 29,3 (40,2) | 2,7× |
| DEIMv2-N | 42,9 | iGPU 15,1 | 2 tuiles 23,9 (42,0) | 1,6× |
| D-FINE-N | 42,8 | NPU+iGPU 15,5 | 2 tuiles 25,9 (42,6) | 1,7× |
| YOLO26-S | 47,7 | iGPU 23,8 | 1 tuile 41,1 (47,7) | 1,7× |
| RF-DETR-N (384 px) | 48,4 | iGPU 23,0 | 2 tuiles 83,4 (48,4) | 3,6× |
| D-FINE-S | 48,5 | NPU+iGPU 34,0 | 2 tuiles 47,0 (48,4) | 1,4× |
| RT-DETRv2-R18 | 48,1 | NPU+iGPU 34,9 | 1 tuile 80,7 (48,0) | 2,3× |
| RF-DETR-S (512 px) | 52,9 | iGPU 45,2 | 2 tuiles 125,7 (53,0) | 2,8× |

Conclusions :
- **Le NPU seul n'est jamais le placement le plus rapide** : il est 1,4 à 3,6 fois plus lent que l'iGPU ou l'hybride. Aucun détecteur actuel n'exploite ce NPU, ce qui confirme l'intérêt de l'objectif.
- **L'INT8 n'est pas un levier pour ces architectures** : plus lent sur les DETR (D-FINE-N : 61,6 ms contre 43,1), nul sur YOLO26-N, −13 % seulement sur YOLO26-S, pour une perte de 0,3 à 2,6 AP. Un petit détecteur est limité par le coût fixe de chaque op, pas par le calcul.
- **Le mode 2 tuiles accélère de 1,3 à 1,7×**, mais il casse la précision de plusieurs modèles, voire le NPU lui-même (YOLO26-S). OptiDet devra être conçu et validé pour rester exact en 2 tuiles.
- **Les backbones ViT (DINOv2, DINOv3) sont à proscrire sur ce NPU** : 3 à 4 fois plus lents que sur l'iGPU.
- **Même un pur CNN comme YOLO26 n'en profite pas** (26 à 29 ms pour 5,4 GFLOPs, alors que ResNet-50 à 224 px tourne en 4 ms). La cible ne s'atteindra pas en réduisant les FLOPs mais en réduisant drastiquement le **nombre d'ops** et en élargissant chacune. M1 doit mesurer ce coût par op.

Cibles des paliers issues de M0 (référence la plus rapide tous placements ÷4 à ÷5) :

| Palier | Référence la plus rapide | Meilleur NPU seul actuel | Cible NPU seul ÷4 / ÷5 |
|---|---|---|---|
| N (≈43 AP) | DEIMv2-N iGPU 15,1 ms | DEIMv2-N 23,9 ms (42,0 AP) | 3,8 / 3,0 ms |
| S (≈48,5 AP) | RF-DETR-N iGPU 23,0 ms | YOLO26-S 41,1 ms (47,7 AP) | 5,8 / 4,6 ms |

Ces cibles demandent d'être 6 à 9 fois plus rapide que le meilleur modèle actuel en NPU seul. Elles sont à confirmer après les micro-benchmarks de M1 (coût par op, débit des convolutions larges en INT8).

### M1 — briques NPU (micro-benchmarks, NPU 3720, batch 1)

Banc `benchmarks/m1/` : 396 blocs, chacun mesuré en FP16 et INT8, sur 1 et 2 tuiles, avec contrôle d'exactitude (1 tuile contre CPU FP32, 2 tuiles contre 1 tuile).

**Modèle de coût**
- **Coût fixe d'une op : faible.** 4 à 6 µs pour une petite convolution, 10 à 30 µs pour une couche de 64 canaux ou moins en 40² ou moins. L'hypothèse de M0 (« trop d'ops ») était fausse : ce sont **certaines ops** qui coûtent.
- **Coût fixe d'un graphe** : 0,5 à 1,5 ms, croissant avec la taille des entrées et sorties (transferts et conversions FP32/FP16).
- **Convolution 3×3 dense + ReLU : 4 à 6 TOPS effectifs par couche** (pic du NPU : 6,1 en FP16, 12,3 en INT8), et l'INT8 la divise par environ 2 dès 128 canaux en 80² ou plus (S7).
- **Réseaux entiers à 640 px, INT8, 2 tuiles** : PlainNet (conv 3×3 + ReLU, 22 GFLOPs) 4,1 ms ; ResNet-18 (30 GFLOPs) 5,5 ms ; ResNet-34 (60 GFLOPs) 9,0 ms, soit 6,7 TOPS. MobileNetV2 7,8 ms (FP16, INT8 impossible), EfficientNet-B0 18,7 ms, ConvNeXt-T 69,9 ms.
- **Résolution** : PlainNet en INT8 passe de 4,1 ms (640) à 2,8 (512), 2,0 (416) et 1,25 (320).

**Règles de conception**

| À privilégier | À éviter | Mesure |
|---|---|---|
| Convolution 3×3 dense, convolution de pas 2 pour réduire | Convolution depthwise (et donc les blocs « mobiles ») | Depthwise : 0,1 à 0,3 TOPS ; INT8 refusé par le compilateur (MobileNetV2/V3, EfficientNet) |
| ReLU (ou LeakyReLU) | SiLU, HardSwish (+30 à 55 % par couche), GELU | S3, S7 |
| BatchNorm (fusionnée, gratuite) | LayerNorm (à 6 400 tokens, presque le coût d'un MLP) | S3, S5 |
| Canaux multiples de 16 | 65, 68, 72 canaux… | S2 (64 et 80 rapides, 68 et 72 lents) |
| Suréchantillonnage : plus proche voisin ou conv transposée 2×2 | Bilinéaire, pixel (un)shuffle | S4 |
| Somme, concaténation + 1×1, résiduel | — | S3, S4 : bon marché |
| Mélange global par convolution | Attention globale au-delà de 400 tokens ; attention par fenêtres | S5 : 1 600 tokens × 256 → attention 8,9 ms contre 2,0 ms pour un mélangeur convolutif |
| Sélection finale côté hôte (0,13 ms pour 2 100 ancres) | `TopK` dans le graphe (1,2 à 2,5 ms) | S5 et mesure CPU |
| Grosses convolutions en INT8 (×2) | INT8 sur des ops limitées par la mémoire (aucun gain) | S6, S7 |

- **Exactitude en 2 tuiles : toutes les ops isolées sont exactes**, attention et `TopK` compris. Les défauts vus en M0 naissent des décisions du compilateur sur un graphe complet, donc **la validation 2 tuiles se fait au niveau du réseau** (critère `npu_lint` à revoir, voir M2).
- **Un NPU « aime » le contraire d'un mobile** : beaucoup de FLOPs dans des ops denses et régulières, plutôt que peu de FLOPs dans des ops irrégulières.

**Premier squelette d'OptiDet (hypothèse de travail pour M2)**
- **Backbone** : de type RepVGG (plusieurs branches à l'entraînement, une seule conv 3×3 + ReLU au déploiement), largeurs multiples de 16, réduction par conv 3×3 de pas 2. Aucune depthwise, squeeze-excitation, SiLU ni LayerNorm.
- **Cou** : PAN léger en conv 3×3 + ReLU, suréchantillonnage au plus proche voisin ou par conv transposée 2×2, fusion par concaténation + 1×1. Au plus une attention globale, au niveau de pas 32 (400 tokens à 640 px, environ 1 ms), si elle apporte de l'AP.
- **Tête** : dense, sans ancres et sans NMS, avec appariement un-à-un (tête un-à-plusieurs auxiliaire à l'entraînement seulement, comme YOLOv10 / YOLO26), et régression des boîtes par distributions (robuste au FP16 et à l'INT8, voir L1). Meilleur score par ancre calculé dans le graphe, top-k sur l'hôte.
- **Entraînement** : distillation depuis un enseignant fort sous licence compatible (D-FINE-L/X, RF-DETR-L), puis QAT INT8. Validation obligatoire en 2 tuiles sur val2017.

**Budget de latence estimé** (INT8, 2 tuiles) :
- **Palier N à 512 px** : backbone de l'ordre de PlainNet-A0 (≈ 14 GFLOPs, 2,8 ms) + cou et tête (≈ 1 ms) ≈ **3,8 ms**. C'est dans la cible (3,0 à 3,8 ms).
- **Palier S à 640 px** : backbone de l'ordre de ResNet-18 (5,5 ms) seul dépasse déjà la cible (4,6 à 5,8 ms), donc il faudra un backbone plus mince ou 512 px, et compter sur la distillation pour l'AP.

Les cibles de M0 sont donc **crédibles pour le palier N, tendues pour le palier S**. L'inconnue principale n'est plus la latence mais **l'AP qu'atteint un réseau aussi « dense et simple »**, ce que M2 doit mesurer.

### M2 — prototype OptiDet

**M2.1 — `npu_lint` fondé sur les détections.** `Detector.lint(images=…)` (CLI : `optinpu lint MODELE --images 30`) mesure le F1 d'appariement des détections CPU FP32 / device (même classe, IoU ≥ 0,7) et exige 0,95. Calibration sur M0 (30 images) :

| Modèle | Perte d'AP réelle en 2 tuiles | Accord 2 tuiles | Accord 1 tuile |
|---|---|---|---|
| RF-DETR-N | 0,0 | 0,990 ✓ | 0,990 |
| D-FINE-N | −0,2 | 0,919 ✗ | 0,994 |
| DEIMv2-N | −0,9 | 0,862 ✗ | 0,975 |
| YOLO26-N | −4,6 | 0,842 ✗ | 0,989 |
| RT-DETRv2 | −3,8 | 0,729 ✗ | 0,990 |

Les deux erreurs de l'ancien critère (RF-DETR faussement signalé, YOLO26-N manqué) disparaissent.

**M2.2 — architecture et latence.** `optinpu/models/optidet.py` : backbone RepVGG (fusion exacte, testée), cou PAN, tête dense un-à-un + un-à-plusieurs, boîtes par distributions. `optinpu/models/optidet_loss.py` : affectation task-aligned (top-10 / top-1), BCE graduée, GIoU, DFL (test de surapprentissage OK). Balayage de 36 candidats (`benchmarks/m2/`, déployés, poids aléatoires), INT8 2 tuiles :

| Candidat | 416 px | 512 px | 640 px | GFLOPs à 512 |
|---|---|---|---|---|
| tiny (32-32-64-128-256), cou 64 | **2,56** | **3,44** | 5,21 | 4,5 |
| a0s (32-48-96-192-384, 1-2-4-8-1), cou 64 | **3,04** | 4,02 | 6,79 | 11,5 |
| a0 (RepVGG-A0, 48-48-96-192-384, 1-2-4-14-1), cou 64 | **3,38** | 4,55 | 7,56 | 15,9 |
| wide (32-64-128-256-512), cou 96 | **3,03** | 4,14 | 7,68 | 14,0 |

- **Les 36 candidats sont exacts en 2 tuiles (FP16 et INT8)** : sans attention ni `TopK`, rien ne casse au découpage. C'est la première fois qu'un détecteur passe ce test sans réserve.
- **Palier N (3,0 à 3,8 ms)** : tenu à 416 px par tous les candidats à cou 64, à 512 px par « tiny ».
- **Le cou et la tête coûtent autant que le backbone** (tiny 512 INT8 : backbone ≈ 1,6 ms, total 3,3 ms). Le format de sortie compte peu (≤ 0,5 ms, sorties FP16 légèrement meilleures). Piste d'optimisation : tronc partagé entre classification et régression.

**M2.4 — premier entraînement d'OptiDet-N** (RepVGG-A0 ImageNet + cou/tête 64, 416 px ; 10 000 images train2017, 10 epochs, batch 16, AdamW 1e-3, bf16 sur iGPU à 11 img/s, 2 h 30 ; `benchmarks/m2/train_optidet.py`).

- **Bugs trouvés et corrigés en route** :
  - biais initial de classification (perte initiale à 3 700) ;
  - normalisation de la perte par la somme des cibles « qualité », minuscule au départ ;
  - perte calculée en bf16 (coordonnées à ±2 px près) ;
  - poids de régression 6 fois trop faible et exposant d'affectation α = 1 au lieu de 0,5 ;
  - bug du moteur L3 : pas de checkpoint sans validation quand la perte dépasse 1.

  Un test de surapprentissage sur 100 vraies images (60 AP en 24 epochs) a validé la chaîne complète.
- **Courbe d'AP (val2017 [0, 500))** : 0,3 → 1,0 → 2,2 → 2,9 → 4,6 → 5,2 → 6,7 → 7,6 → 8,3 → **8,5**. Les deux têtes progressent au même rythme (le retard n'est pas propre à la tête un-à-un).
- **Déploiement du modèle entraîné** :

| | AP | Latence NPU (2 tuiles) | Accord détections vs CPU |
|---|---|---|---|
| CPU FP32 | 8,5 | — | — |
| NPU FP16 | 8,5 | 4,80 ms | 1,000 |
| NPU INT8 (PTQ, 300 images) | 7,2 | **3,36 ms** | 1,000 |

- **Ce qui est établi** : l'architecture tient le budget du palier N (3,36 ms en INT8) et elle est **exacte sur le NPU en 2 tuiles** (accord parfait) ; la chaîne d'entraînement est correcte ; la quantification PTQ coûte 1,3 AP, que la QAT devrait récupérer.
- **Ce qui ne l'est pas** : l'AP finale atteignable. 8,5 AP après environ 0,85 epoch de COCO ne dit presque rien des 43 AP visés : les références s'entraînent sur 300 à 500 epochs de COCO complet. L'estimation « 15 à 25 AP » donnée avant le run était trop optimiste.

**M2.5 — distillation et QAT** (même budget que M2.4 : 10 000 images, 10 epochs ; val2017 [0, 500)).

- **Distillation PKD** depuis l'encodeur hybride de D-FINE-S (Apache-2.0, 256 canaux aux pas 8 / 16 / 32, alignés sur le cou à 416 px). Perte de corrélation de Pearson entre cartes standardisées, poids 2,0, adaptateurs 1×1 réservés à l'entraînement (retirés par `deploy()`, ignorés par `from_checkpoint`). API : `OptiDet.enable_distillation(DFineTeacher())`, script `train_optidet.py --distill ustc-community/dfine-small-coco`.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Sans distillation (M2.4) | 0,3 | 1,0 | 2,2 | 2,9 | 4,6 | 5,2 | 6,7 | 7,6 | 8,3 | 8,5 |
| Avec distillation | 0,2 | 1,5 | 2,8 | 3,8 | 4,6 | 6,8 | 7,6 | 8,8 | 10,0 | **10,0** |

  **+1,5 AP (+18 %), AP50 16,1 contre 13,9**, pour 21 min par epoch au lieu de 14,5 (l'enseignant tourne à chaque batch sur l'iGPU). L'écart se creuse au fil des epochs.
- **Où se perd l'INT8 en PTQ** (`benchmarks/m2/ptq_variants.py`, checkpoint M2.4, FP32 8,4 AP sur CPU) :

| Variante PTQ | AP INT8 (CPU) |
|---|---|
| Défaut NNCF, preset MIXED ou 1 000 images de calibration | 7,1 |
| Correction de biais complète | 3,9 |
| Tête, sorties, stem, premier étage ou cou gardés en FP16 | 7,1 à 7,2 |
| Backbone entier, ou ses étages 2 à 4, gardés en FP16 | **8,1** |

  La perte vient du **backbone RepVGG fusionné** : la fusion des branches 3×3, 1×1 et identité donne des poids aux distributions larges, mal servies par l'INT8 par canal (problème décrit par QARepVGG). Garder ces étages en FP16 coûte environ 28 % de latence : rejeté.
- **QAT** (`benchmarks/m2/qat.py`) : `nncf.quantize` sur le modèle fusionné insère des quantificateurs simulés dans `forward()` ; `OptiDet.loss()` passe par `forward()` pour qu'ils agissent pendant l'entraînement (test `test_qat_fake_quantizers_act_in_the_loss_path`). 2 epochs, AdamW 1e-4, FP32 (6,6 img/s, 51 min) sur le checkpoint distillé, puis export INT8 :

| Checkpoint distillé (FP32 : 10,0 AP) | AP | Latence NPU (2 tuiles, machine au repos) | Accord détections vs CPU |
|---|---|---|---|
| INT8 PTQ (point de départ) | 8,5 | 3,48 ms | — |
| INT8 après QAT, CPU | 9,7 | — | — |
| **INT8 après QAT, NPU** | **10,0** | **3,37 ms** | 0,964 (lint OK) |

  **La QAT récupère toute la perte INT8** et ne ralentit pas le graphe : 54 FakeQuantize contre 53 en PTQ OpenVINO, aux mêmes endroits (`compare_int8_graphs.py`). Les 4,15 ms mesurés en fin de run venaient de la contention avec l'évaluation.
- **Intégration dans la librairie** : `Detector.train_qat(data, val)` (CLI : `optinpu qat MODELE --checkpoint best.pt --train …`) fusionne le modèle, insère et calibre les quantificateurs (300 images d'entraînement), entraîne avec les réglages ci-dessus (`QAT_DEFAULTS`) puis exporte l'INT8 directement depuis les quantificateurs entraînés. Après QAT, `int8` devient la précision par défaut et la seule exportable. Le checkpoint contient la configuration NNCF (`nncf.torch.get_config`) : `from_checkpoint` restaure les quantificateurs sans recalibrer. Réservé aux modèles dotés de `fuse()` dont la perte passe par `forward()` (OptiDet).
- **Bilan M2** : OptiDet-N tient le palier N en latence (**3,37 ms en INT8, NPU seul, exact en 2 tuiles**) et la recette « distillation puis QAT » est validée : +18 % d'AP, puis zéro perte au passage en INT8. L'AP absolue reste inconnue : 10 AP après 0,85 epoch de COCO ne permet pas d'extrapoler aux 43 AP visés, c'est l'objet de M3.
- **Pistes pour M3** :
  - entraînement long sur COCO complet avec distillation dès le départ et QAT en fin de recette ;
  - enseignant plus fort (D-FINE-L/X) ;
  - blocs QARepVGG, pour rendre la PTQ presque sans perte et alléger la QAT ;
  - tronc partagé dans la tête (voir M2.2).

Enseignements pour OptiDet (M0-M2) :
- **La référence à battre est le meilleur placement tous confondus** (iGPU ou hybride selon le modèle, voir M0) : OptiDet doit faire mieux en NPU seul.
- **D-FINE n'est pas limité par le calcul sur le NPU** mais par le nombre de petites ops et les faibles nombres de canaux. Le backbone (16 à 32 canaux à 320×320) et le décodeur (300 requêtes, `integral` 300×4×33, sommes de l'attention déformable) sont les points chauds. L'INT8 ajoute des quantize/dequantize sans réduire ce surcoût.
- À l'inverse, **un CNN large et régulier (ResNet-50) gagne ×2,4 en INT8.** OptiDet doit viser des ops peu nombreuses, larges et alignées (canaux multiples de 32 ou 64), avec un décodeur compact, pour exploiter l'INT8.
- Sur ce portable, le CPU en INT8 (49 ms) rivalise avec le NPU en FP16 pour D-FINE-S : le NPU n'est intéressant que pour des modèles adaptés à lui. C'est la justification même d'OptiDet.
- **Le raffinement des boîtes par distributions (D-FINE FDR) résiste au bruit FP16** du NPU (erreur de 0,025 sur les boîtes), contrairement à la régression directe de RT-DETRv2 (0,17). C'est un argument pour une tête de type D-FINE.
- Tout ce qui ne dépend que de la taille d'entrée doit être une constante au moment de l'export.

### P1 — profileur mesuré (en cours)

`optinpu.profiler.profile_measured` : variantes du graphe mesurées dans un sous-processus, côte à côte et en round-robin avec le modèle complet (`benchmarks/p1/`). Bruit sur YOLO26-N : ±0,32 ms pour 25,9 ms (±1,2 %).

**P1.0 — la latence dépend-elle des valeurs ?** (`benchmarks/p1/weights.py`, 7 tours de 40 inférences par variante)

| Modèle | Poids aléatoires denses | Entrées nulles | 50 % des poids à zéro | 90 % à zéro |
|---|---|---|---|---|
| ResNet-50 (640) | −0,3 % | −0,4 % | **−25,5 %** | **−35,3 %** |
| MobileNetV2 (640) | +1,5 % | +0,6 % | −2,6 % | −3,8 % |
| YOLO26-N FP16 | −5,2 % | +0,1 % | −5,9 % | −8,6 % |
| YOLO26-N INT8 | −3,7 % | 0,0 % | −6,3 % | −6,5 % |
| D-FINE-N FP16 | +0,1 % | **+11,0 %** | −2,8 % | −2,8 % |
| OptiDet-N INT8 (QAT) | −2,3 % | −2,2 % | **−15,3 %** | **−21,7 %** |

- **La valeur des poids denses ne compte pas** (écarts dans le bruit, sauf sélection dépendante des données) : mesurer une variante avec des poids aléatoires denses donne la latence du modèle entraîné. C'est la prémisse des what-if.
- **Les ops dépendantes des données font exception** : sélection TopK (YOLO26, quelques %) et surtout D-FINE (+11 % avec des entrées nulles : sélection des requêtes, échantillonnage de l'attention déformable). Pour ces modèles, profiler sur de vraies images.
- **Le NPU exploite la parcimonie non structurée des poids** : jusqu'à −35 % sur ResNet-50, −22 % sur OptiDet-N INT8 ; peu d'effet sur les réseaux à depthwise ou dominés par d'autres ops. Nouveau levier (élagage + fine-tuning), absent du plan initial.

**P1.1 — premier profil : YOLO26-N FP16, 2 tuiles** (25,9 ms)
- **La tête coûte 72 % de la latence** (18,6 ms, profil en profondeur). Compteurs du constructeur : la sélection finale (`Divide` 19,5 ms, `GatherElements` 15,4 ms, `TopK` 4,1 ms, durées qui se chevauchent) tourne sur les SHAVE (DSP du NPU : 45,9 ms cumulés, contre 18,9 ms sur les DPU de convolution). → réécriture P2.
- Retirer les 87 SiLU ferait gagner 2,6 ms (10 %) : borne haute du gain de SiLU → ReLU.
- **Le profil par préfixes n'est pas additif** sur ce NPU : segments négatifs jusqu'à −3,7 ms, le compilateur prenant d'autres décisions sur un graphe tronqué. Le contournement de régions est plus fiable.
- Contourner la Sigmoid de la tête fait **perdre le NPU** (`ZE_RESULT_ERROR_DEVICE_LOST`), comme YOLO26-S en M0 : le bug touche la sélection finale selon les valeurs qu'elle reçoit. Le sous-processus l'a isolé.

**P1.2 — validation : qui prédit ce que rapporte un changement ?** (`benchmarks/p1/validate.py` et `summary.py`, 6 modèles, 194 interventions, PC utilisé en parallèle, priorité basse)
- **Vérité terrain** : gain mesuré de chaque intervention (contourner une région ou un type d'op, SiLU → ReLU, LayerNorm retirée, sélection sur CPU), mesuré deux fois dans un ordre inverse ; vérité = moyenne des deux.
- **Prédicteurs** : compteurs du constructeur sommés sur les ops retirées (bruts, ou rapportés à la latence) ; modèle additif à la nn-Meter (chaque op seule sur le NPU, moins le coût d'un graphe réduit à ses entrées) ; et notre propre mesure répétée (run 2 contre run 1 : la limite fixée par notre bruit, comparaison défavorable à notre méthode).
- **Regret** : part du meilleur gain possible perdue en appliquant le premier choix du prédicteur.

Interventions significatives (gain > 2 × bruit), moyenne sur 5 modèles :

| Prédicteur | Corrélation de rang | Erreur (% de la latence) | Top 3 | Regret |
|---|---|---|---|---|
| Compteurs du constructeur (rapportés à la latence) | +0,49 | 4,8 % | 0,60 | 0,29 |
| Modèle additif (ops isolées) | +0,34 | 17,6 % | 0,87 | 0,30 |
| **Mesure répétée** | **+0,76** | **1,7 %** | **0,87** | **0,02** |

- Par modèle, les compteurs vont de +0,01 (YOLO26-N) à +0,86 (D-FINE-N), le modèle additif de −0,28 (ResNet-50) à +0,75 (EfficientNet-B0). Même avec le bon top 3, le premier choix des compteurs ne rapporte que 38 % du meilleur gain sur D-FINE-N et 18 % sur EfficientNet-B0.
- **Les sommes n'ont pas de sens physique** : compteurs 126 à 349 ms et modèle additif 101 à 245 ms pour 18 à 31 ms de latence réelle. Les compteurs ne sont pas non plus stables : 66 ms puis 126 ms de somme pour YOLO26-N d'un run à l'autre.
- **Limite : les petits modèles.** Le bruit (0,1 à 0,3 ms) dépasse la plupart des gains sur MobileNetV2 (7,5 ms, 3 interventions significatives) et OptiDet-N (4,0 ms, aucune ; corrélation de la mesure répétée −0,09).

**P1.3 — répétitions adaptatives.** Chaque tour mesure tous les modèles du lot dans un ordre tiré au hasard ; le gain d'une variante est la moyenne des écarts tour par tour avec le modèle complet, avec un intervalle de confiance à 95 %. Les tours continuent jusqu'à une demi-largeur de 0,5 % de la latence (au moins 0,02 ms), dans la limite de 40 tours (`profiler.Adaptive`, `paired_saving` ; utilisé par le profileur, l'optimiseur et la validation). « Significatif » = gain positif avec 95 % de confiance. MobileNetV2 et OptiDet-N remesurés (PC toujours utilisé en parallèle ; 40 tours atteints à chaque lot) :

| | MobileNetV2 avant → après | OptiDet-N avant → après |
|---|---|---|
| Interventions significatives | 3 → 11 | 0 → 12 |
| Mesure répétée : corrélation (toutes) | +0,48 → +0,88 | −0,09 → +0,45 |
| Mesure répétée : erreur moyenne | 0,34 → 0,08 ms | 0,18 → 0,06 ms |

Bilan sur les 6 modèles, interventions significatives :

| Prédicteur | Corrélation de rang | Erreur (% de la latence) | Top 3 | Regret |
|---|---|---|---|---|
| Compteurs du constructeur (rapportés à la latence) | +0,56 | 3,4 % | 0,56 | 0,26 |
| Modèle additif | +0,45 | 11,1 % | 0,78 | 0,40 |
| **Mesure répétée** | **+0,68** | **1,0 %** | **0,83** | **0,00** |

- **Pour décider, la mesure gagne partout** : regret nul sur les 6 modèles, contre 0,26 (compteurs) et 0,40 (additif) en moyenne ; meilleure erreur absolue partout.
- **Sur les petits modèles, l'ordre fin des gains moyens reste au niveau du bruit** : corrélation de la mesure répétée +0,62 (MobileNetV2) et +0,21 (OptiDet-N) contre +0,88 et +0,52 pour les compteurs, sur des gains significatifs mais proches les uns des autres. La limite de 40 tours est atteinte à chaque lot : il faudra plus de tours et une machine au repos pour l'article.
- **Critère de sortie de P1 atteint** pour ce qui sert à décider (top 3, regret, erreur absolue) ; campagne finale de l'article à refaire sur machine au repos.

### P2 — réécritures (en cours)

**Sélection finale sur le CPU** (`graph.split_from` : la première op TopK / NMS et tout ce qui en dépend ; `benchmarks/p2/selection_tail.py`, latences côte à côte, AP sur val2017 [0, 500)) :

| Modèle | Placement | Latence | Accord vs CPU | AP |
|---|---|---|---|---|
| YOLO26-N FP16 | CPU FP32 (référence) | — | — | 44,8 |
| | NPU seul, 2 tuiles | 25,5 ms | 0,842 | 40,7 |
| | NPU seul, 1 tuile | 28,6 ms | — | — |
| | **NPU 2 tuiles + sélection sur CPU** | **8,6 ms** | **0,996** | **44,8** |
| | NPU 1 tuile + sélection sur CPU | 12,5 ms | 0,996 | 44,8 |
| YOLO26-N INT8 | NPU 2 tuiles + sélection sur CPU | 8,2 ms | 0,509 | 24,9 |
| | NPU 1 tuile + sélection sur CPU | 12,6 ms | 0,923 | 42,3 |
| YOLO26-S FP16 | CPU FP32 (référence) | — | — | 50,8 |
| | NPU seul, 2 tuiles | 32,6 ms | perte du NPU (M0) | — |
| | NPU seul, 1 tuile | 40,7 ms | — | — |
| | **NPU 2 tuiles + sélection sur CPU** | **15,7 ms** | **1,000** | **50,7** |

- **Une réécriture exacte de 16 ops rend YOLO26-N 3× plus rapide, YOLO26-S 2,1× (2,6× face au meilleur placement NPU seul de M0)**, et les deux battent désormais l'iGPU (M0 : 10,9 et 23,8 ms).
- **Elle corrige aussi les défauts du mode 2 tuiles** : la perte silencieuse de 4,1 AP de YOLO26-N et la perte du NPU de YOLO26-S (500 images sans erreur). Ces deux défauts naissent de la sélection compilée avec le reste du graphe : la sélection seule, sur 2 tuiles, ne plante pas (`tail_repro.py`). Rapport Intel mis à jour.
- En INT8, le mode 2 tuiles reste faux même sans la sélection (AP 24,9) : le défaut est alors dans le corps du réseau, et l'INT8 n'accélère pas YOLO26-N (8,2 contre 8,6 ms). La garde d'exactitude doit refuser cette combinaison.
- Pour les DETR, le premier TopK est au milieu du graphe (sélection des requêtes) : ce qui en dépend, c'est tout le décodeur (300 à 500 ops). Le candidat correspondant est le placement hybride (décodeur sur iGPU ou CPU), pas une simple sélection sur l'hôte.

**Optimiseur** (`optinpu.optimize`, CLI `optinpu optimize` ; `benchmarks/p2/optimize_model.py`). Boucle gloutonne sur un état de déploiement (graphe, options de compilation, placement de la sélection) ; garde d'exactitude sur 30 images de train2017, AP rapportée sur val2017 [0, 500). Réécritures actuelles : sélection ou décodeur hors du NPU, nombre de tuiles, depthwise en conv dense (exactes) ; SiLU / HardSwish / GELU → ReLU, LayerNorm → BatchNorm, élagage 50 / 75 % (recommandations, latence mesurée).

YOLO26-N FP16, sans intervention : **26,0 → 8,55 ms (3,04×)**, accord 0,966 → 1,000, AP 44,8 (= CPU). Rejetés : depthwise en dense (aucun gain), 1 tuile (12,7 ms). Recommandé : SiLU → ReLU, encore −25 % (6,4 ms) après réentraînement ; l'élagage n'aide pas ce modèle.

- **Limite de la garde** : sur ces 30 images de train2017, le modèle d'origine en 2 tuiles atteint un accord de 0,966 (au-dessus du seuil de 0,95), alors qu'il perd 4,1 AP sur val2017 (accord de 0,842 sur 30 images de val). 30 images ne suffisent pas toujours : à étudier (nombre d'images, seuil, accord par classe).

### P3 — réécritures approchées (en cours ; P3.1 à P3.3 faits, P3.4 en cours)

**P3.1 — INT8 dans l'optimiseur, budget d'AP.** Candidat « approché » : quantification NNCF du graphe courant (`rewrites.int8_ptq`, `quantize.quantize_model`), calibrée sur des échantillons passés par la tâche. Avec un jeu annoté (`optimize(..., evaluation=CocoSubset, ap_budget=0.005)`), un candidat approché doit garder l'AP à moins de 0,5 point de l'original sur CPU FP32, mesurée dans un sous-processus (`sandbox.run_evaluation`). Protocole (`benchmarks/p2/optimize_model.py`, résultats dans `runs/p3/`) : garde sur 30 images de train2017, calibration sur 300 autres, budget d'AP sur train2017 [0, 200), rapport sur val2017 [0, 500).

| Modèle | Avant | Après | Appliqué | AP val2017 : CPU / avant / après |
|---|---|---|---|---|
| YOLO26-N FP16 | 25,9 ms (accord 0,966) | **8,5 ms (3,04×)**, accord 1,000 | sélection sur CPU | 44,8 / 40,7 / 44,8 |
| D-FINE-N FP16 | 25,4 ms (accord 0,925) | **15,0 ms (1,69×)**, accord 0,991 | décodeur sur iGPU | 45,8 / 45,6 / 45,6 |

- YOLO26-N : l'INT8 (schéma CNN) gagne 0,65 ms (7,9 ms) mais tombe à 28,8 AP sur le jeu de budget (seuil 48,1) : **rejeté par le budget d'AP**, à juste titre (l'INT8 en 2 tuiles est faux sur ce modèle, P2). Le schéma « transformer » ne compile pas. SiLU → ReLU reste la meilleure piste : −26 % (6,3 ms) après réentraînement.
- D-FINE-N : l'optimiseur retrouve seul le placement hybride de L4. L'INT8 compile mais coûte +10,8 ms : rejeté (confirme L2). Recommandations : élagage 75 % −4,7 %, LayerNorm → BatchNorm −3,0 %, SiLU → ReLU sans effet.
- **L'AP détecte ce que l'accord sur 30 images laisse passer** : YOLO26-N d'origine en 2 tuiles, accord 0,966 mais AP 44,6 contre 48,6 sur CPU (train2017 [0, 200)). Quand des annotations existent, l'AP devrait aussi décider si l'état de départ est exact (à faire en P3.3).
- **Échec de compilation INT8 (`benchmarks/p3/int8_compile.py`)** : le schéma « transformer » de NNCF produit sur YOLO26-N un `FakeQuantize` à paramètres de 2 canaux sur un tenseur de 128 canaux, que le compilateur NPU refuse (IR FP16 comme FP32) ; le schéma CNN compile (tête NPU 8,2 ms contre 8,9 en FP16). L'optimiseur essaie désormais les deux schémas sur les graphes avec attention ou LayerNorm et laisse la mesure trancher.

**P3.2 — précision mixte guidée par la latence.** Déclenchée quand l'INT8 complet est significativement plus rapide mais hors budget d'AP : chaque groupe de modules contenant des convolutions / matmuls (`rewrites.quantizable_groups`, ~10 groupes) est gardé seul en flottant (`int8_ptq(..., keep_float=...)`, `IgnoredScope` de NNCF) ; on mesure l'AP qu'il récupère et la latence qu'il coûte face à l'INT8 complet ; les groupes sont choisis par AP récupérée par ms (`choose_float_groups`, additivité supposée), puis le mélange est vérifié (latence appariée, AP) en ajoutant des groupes jusqu'à passer le budget. Un groupe sensible mais coûteux (plus d'un quart du gain de l'INT8) est découpé en sous-modules (`split_group`, ajouté après le run OptiDet). **Protocole changé** : les modèles ont été entraînés sur train2017, les décisions se prennent désormais sur val2017 (garde [4970, 5000), calibration [4500, 4800), budget d'AP [2500, 2700)), le rapport sur val2017 [0, 500). Un original qui plante sur le NPU devient un état à réparer (accord 0) au lieu d'arrêter l'optimiseur.

| Modèle | Avant | Après | Appliqué | AP val2017 : CPU / avant / après |
|---|---|---|---|---|
| OptiDet-N flottant (avant QAT) | 4,52 ms | 4,41 ms (1,03×) | INT8 mixte, backbone et tête en flottant | 9,98 / 10,00 / 10,00 |
| YOLO26-S FP16 | 33,0 ms, **perte du NPU** | **13,7 ms (2,40×)**, accord 0,973 | sélection sur CPU + INT8 mixte (6 groupes sur 18 en flottant) | 50,8 / plante / **50,6** |

- **YOLO26-S** : l'INT8 complet (12,8 ms) perd 10,4 AP sur le jeu de budget (36,1 contre 46,5). La perte se concentre dans **les deux blocs d'attention** (`model.10` : +6,9 AP s'il reste en flottant, pour +0,08 ms ; `model.22` : +1,8 AP) et dans la tête (+1,1 AP) et le début du réseau (`model.0`, `model.1` : +0,8 et +1,0 AP). Avec 5 groupes en flottant, AP 46,0 (juste sous le seuil) ; avec 6, AP 46,6 : accepté. **13,7 ms contre 23,8 ms sur l'iGPU (M0) et 41,1 ms pour le meilleur placement NPU seul de M0**, pour 0,2 point d'AP en moins. Recommandation : SiLU → ReLU, encore −25 % (10,3 ms) après réentraînement.
- **OptiDet-N** : l'INT8 complet (3,05 ms) perd 1,2 AP ; la perte est dans le backbone (+0,95 AP en flottant, mais +1,22 ms) et la tête (+0,26 AP, +0,05 ms). Le mélange accepté ne gagne que 2,4 % : la précision mixte sans réentraînement ne rivalise pas avec la QAT de M2.5 (3,37 ms, sans perte). Le découpage du backbone en étages (ajouté depuis) devrait faire mieux, à mesurer. Recommandation : élagage à 75 %, −21,6 %.
- Coût de la recherche : ~70 min pour YOLO26-S (18 quantifications sur CPU à 640 px, 18 évaluations d'AP), ~15 min pour OptiDet-N.

OptiDet est retiré du périmètre de l'article (décision du 2026-10-29) : le banc de l'article est celui des 9 modèles tiers de M0 et des CNN.

**P3.3 — livrable, CLI, exactitude par l'AP.**
- **Livrable** : `OptimizationReport.save(dossier)` écrit `report.md` (tableaux lisibles), `report.json` et le déploiement optimisé (`part0.xml`, `part1.xml` pour une queue déportée, `deployment.json` : appareils, options de compilation, changements) ; `optimizer.load_deployment(dossier)` le recharge pour `tasks.Deployed` ou `profiler.measure`.
- **CLI** : `optinpu optimize MODELE --images GARDE/ --calibration CALIB/ --annotations JSON IMAGES/ --ap-budget 0.5 --out DOSSIER/` (+ `--task`, `--letterbox`, `--imagenet-norm` pour un fichier IR / ONNX, `--no-int8`).
- **Exactitude par l'AP** : avec un jeu annoté, l'original (et toute réparation) n'est « exact » que si son AP reste dans le budget, en plus de l'accord des détections. Corrige le cas YOLO26-N (accord 0,966, −4 AP).
- **Précision mixte moins chère** : la sensibilité des groupes est estimée avec 64 images de calibration au lieu de 300 (~5× moins de CPU) ; le mélange retenu est quantifié avec toutes et vérifié. Une restauration du flottant par chirurgie du graphe INT8 a été écartée : NNCF compresse déjà les poids en int8 en mémoire.
- README : section « Profile and optimize your own model ».
- **Validation de bout en bout (P3.3)** (`benchmarks/p3/cli_folders.py`, dossiers d'images comme un utilisateur, `--no-int8`) sur YOLO26-N : l'original est jugé inexact par l'AP (32,4 contre 36,4 sur CPU, accord 0,819) ; la sélection sur CPU est acceptée après vérification de l'AP (36,6) ; 25,8 → 8,45 ms (3,05×). Le déploiement rechargé (`npu+cpu`) donne un accord de 0,995 avec le CPU FP32 sur 20 autres images.

**P3.4 — banc des détecteurs (par paquets).** Résultats dans `runs/p3/optimize_*.json`, AP rapportée sur val2017 [0, 500).

| Modèle | Avant | Après | Appliqué | AP val2017 : CPU / avant / après | Paquet |
|---|---|---|---|---|---|
| YOLO26-N FP16 | 26,2 ms (−3,4 AP) | **8,87 ms (2,96×)** | sélection sur CPU | 44,8 / 40,7 / 44,8 | 2 |
| YOLO26-S FP16 | 33,0 ms, perte du NPU | **13,7 ms (2,40×)** | sélection sur CPU + INT8 mixte | 50,8 / plante / 50,6 | P3.2 |
| D-FINE-N FP16 | 26,0 ms (−0,25 AP, borne −0,64) | **15,4 ms (1,69×)** | décodeur sur iGPU | 45,8 / 45,6 / 45,6 | 3 |
| D-FINE-S FP16 | 47,0 ms (accord 0,909) | **34,7 ms (1,36×)** | décodeur sur iGPU | 51,6 / 51,2 / 51,5 | 1 |
| DEIMv2-N FP16 | 23,9 ms (−0,72 AP) | **13,1 ms (1,82×)** | décodeur sur iGPU | 46,7 / 45,7 / 46,6 | 3 |
| RT-DETRv2-R18 FP16 | 47,5 ms (−2,9 AP) | **29,0 ms (1,64×)** | coupe à `model.enc_output` vers l'iGPU + INT8 (schéma transformer) | 50,8 / 48,0 / 50,7 | 3 (paquet 2 : 43,3 ms) |
| RF-DETR-N FP16 | 82,9 ms | **37,4 ms (2,22×)** | décodeur sur iGPU + INT8 mixte (schéma transformer, 3 petites têtes en flottant) | 51,4 / 51,3 / 51,1 | 2 |

Corrections de méthode en cours de route (toutes avant d'être appliquées au paquet suivant) :
1. **Budget d'AP jugé sur 200 images trompeur** (paquet 1) : la précision mixte de YOLO26-N passait à −0,3 point sur 200 images mais perdait 0,9 point sur les 500 du rapport. Désormais : 500 images (val2017 [2500, 3000)), AP calculée sur 10 tranches de 50 images pour le candidat et pour le CPU FP32, **test de non-infériorité apparié** (perte inférieure au budget au seuil de 5 %), aussi pour juger l'original et les réparations. Avec lui, le mélange de YOLO26-N (−0,37 point, borne à −1,2) est refusé et la sélection sur CPU acceptée (+0,01 ± 0,08).
2. **Test bilatéral trop strict** (paquet 2) : la question « la perte reste-t-elle sous le budget ? » est unilatérale. Le seuil est désormais `t(0,95)` au lieu de `t(0,975)`. Règle fixée pour la suite et pour l'article. RT-DETRv2 + INT8 (34,7 ms, −0,19 point) aurait été accepté avec cette règle ; il sera refait au paquet 3.
3. **Candidats approchés essayés sur un état inexact** (paquet 2, RT-DETRv2) : temps perdu dans une recherche de précision mixte sans objet. Désormais, les candidats approchés attendent que l'état soit réparé. (Remplacée au passage au NPU seul, ci-dessous : ils peuvent réparer, mais sans recherche de précision mixte sur un état inexact.)
4. **Coupe hybride au TopK sous-optimale pour les DETR** : RT-DETRv2 décodeur sur iGPU à 43,3 ms contre 33,9 ms en L4, où la coupe était à la fin de l'encodeur (les têtes de l'encodeur et une LayerNorm sur 8 400 tokens restent sinon sur le NPU). Ajout de candidats de coupe aux frontières des grands modules avant la sélection (`optimizer._boundaries` : `enc_output`, `enc_score_head`…).
5. **Quantificateurs hors groupes jamais remis en flottant** (paquet 3, D-FINE-N) : `quantizable_groups` ajoute un groupe `(rest)`, les ops des régions sans op lourde, évalué et choisi comme les autres. Sur D-FINE-N : 17 groupes dont `(rest)` (182 ops) ; tous en flottant, il ne reste plus aucun FakeQuantize (64 avant). À relancer : D-FINE-N, DEIMv2-N, D-FINE-S.

Observations :
- **L'INT8 corrige le défaut 2 tuiles de RT-DETRv2** : en INT8 (schéma transformer), l'original en 2 tuiles n'est plus qu'à −0,29 point du CPU, contre −2,9 en FP16. Le compilateur prend d'autres décisions de découpage de l'attention.
- **D-FINE-S** : l'INT8 complet gagne 5,6 ms mais perd 6 points ; garder 7 groupes sur 16 en flottant n'en récupère que 2. Lu d'abord comme une perte diffuse ; le paquet 3 montre plutôt une limite de l'outil (ci-dessous).
- **Limite de la précision mixte : les quantificateurs hors groupes (paquet 3).** Sur D-FINE-N, l'INT8 complet perd 5,7 points et garder **18 groupes sur 20** en flottant laisse encore −3,1 points. Vérification sur CPU : avec tous les groupes en flottant, il reste **64 FakeQuantize**, presque tous sur des ops de premier niveau du décodeur (`decoder/aten::{select,mul,add,sub,stack}`, l'arithmétique des points de référence et des boîtes), que `quantizable_groups` ne couvre pas, faute d'op lourde dans ces régions. La perte vient très probablement de là, pas des modules. Probablement la même cause pour DEIMv2-N (INT8 −6,5 points ; 8 groupes sur 11 en flottant laissent −0,43 point, borne −1,1) et pour D-FINE-S. Corrigé (correction 5).
- **RT-DETRv2 (paquet 3)** : la coupe aux frontières trouve `model.enc_output` (37,4 ms, AP +0,00 point), puis l'INT8 transformer sur cet état passe le test de non-infériorité de justesse (−0,20 point, borne −0,50) : 29,0 ms, −0,07 point sur le rapport. L'INT8 CNN perd 2,6 points.
- **La précision mixte peut accélérer un état hybride** : sur DEIMv2-N et D-FINE-N, une partie du décodeur tourne sur l'iGPU ; garder `decoder.decoder` / `decoder.layers` en flottant fait **gagner** 0,35 à 0,87 ms (les FakeQuantize coûtent sur l'iGPU), et l'INT8 transformer y coûte +10,8 à +11 ms.
- **Coupe aux frontières vs coupe au TopK** : seule RT-DETRv2 profite d'une coupe plus haute ; sur D-FINE-N et DEIMv2-N, le décodeur sur iGPU (coupe au premier TopK) reste le meilleur.
- **RF-DETR-N : la précision mixte n'est pas additive sur ce NPU.** Avec le schéma CNN (−4,2 points), l'affinage descend jusqu'aux 12 couches du backbone DINOv2 : garder **toute** la pile d'encodeurs en flottant coûte +4,2 ms, mais garder **une seule** couche du milieu (1 à 3) en flottant au milieu de couches INT8 coûte +13 à +16 ms (la couche 0 seule : +0,18 ms). Les conversions de précision et le découpage en tuiles entre couches de précisions différentes coûtent plus que le calcul. Les modèles de coût additifs des méthodes de précision mixte (HAQ…) supposent le contraire ; seule la mesure du mélange complet est fiable. Le schéma transformer, lui, ne perd que 0,4 point : le mélange retenu (3 petites têtes en flottant) est à −0,17 point.
- Coût : ~2 h 30 pour le paquet 2, dont ~1 h 15 pour RF-DETR-N (trois niveaux d'affinage, 20 groupes évalués sur 500 images). ~2 h 20 pour le paquet 3 (RT-DETRv2 45 min, DEIMv2-N 35 min, D-FINE-N 1 h).

**Passage au NPU seul (2026-10-29, après le paquet 3).** Les résultats des paquets 1 à 3 sont déplacés dans `runs/p3/offload-gpu-cpu/` : ils serviront de comparaison (P5). Changements de l'optimiseur :
- `optimize(..., offload=())` par défaut : seule une sélection finale légère va sur le CPU ; décodeur et coupes aux frontières seulement vers les appareils de `offload` (`--offload gpu` dans la CLI et le banc, résultats dans `runs/p3/offload-<appareils>/`).
- **Les candidats approchés peuvent réparer** un état inexact (jugés par l'AP contre le CPU FP32, comme toujours), puisque l'INT8 supprime le défaut 2 tuiles de RT-DETRv2 ; la recherche de précision mixte reste réservée aux états exacts. Cela remplace la correction 3.
- **Nouvelle réparation exacte sans quitter le NPU : `decoder-on-1-tile`**, le décodeur compilé à part sur 1 tuile, le reste sur 2 (`State.offload_config`). Mesuré sur RT-DETRv2 FP16 (30 images de garde) :

| RT-DETRv2 FP16 sur NPU | Latence | Accord |
|---|---|---|
| original, 2 tuiles | 47,1 ms | 0,721 |
| découpé au TopK, les deux parties sur 2 tuiles | 56,0 ms | 0,721 |
| tête sur 1 tuile, décodeur sur 2 | 72,3 ms | 0,721 |
| **décodeur sur 1 tuile, tête sur 2** | 75,4 ms | **0,995** |
| tout sur 1 tuile | 80,6 ms | 0,995 |

  Le défaut est bien dans le décodeur, mais la réparation est chère : découper le graphe en deux inférences NPU coûte à lui seul +8,9 ms, et le décodeur sur 1 tuile +19 ms. Elle ne bat `tiles=1` que de 5 ms ; la mesure départagera avec l'INT8 transformer en 2 tuiles.

### P3.5 — banc en NPU seul (en cours)

**Modèles de l'article (décision du 2026-10-30)** : 7 détecteurs, YOLO26-N/S, D-FINE-N/S, DEIMv2-N, RT-DETRv2-R18, RF-DETR-N (DEIMv2-S retiré : export à −4 AP ; RF-DETR-S retiré : 125 ms, redondant avec RF-DETR-N), et **3 classifieurs ImageNet** pour montrer l'outil hors détection : ResNet-50, MobileNetV2, EfficientNet-B0 (poids torchvision, 224 px).

**Paquets** : (1) RT-DETRv2, D-FINE-N, DEIMv2-N ; (2) D-FINE-S, RF-DETR-N, YOLO26-N ; (3) YOLO26-S et les 3 classifieurs.

**Classifieurs dans la lib** : `tasks.ClassificationTask` (prétraitement torchvision : petit côté redimensionné, recadrage central, normalisation ; identique à torchvision à 1e-5, testé), accord = part des images de même classe top-1 ; `data.imagenetv2_subset` (ImageNetV2 matched frequency, 10 000 images, une permutation fixe découpée en tranches disjointes), `evaluate.evaluate_top1`, job `evaluate` du sandbox et optimiseur indépendants de la métrique (`score` / `fold_scores`, `task.metric` = AP ou top-1). Protocole : garde [7300, 7600) (300 images), calibration [7000, 7300), budget top-1 [5000, 7000) en 10 plis de 200, rapport [0, 5000). Top-1 CPU FP32 sur 500 images : ResNet-50 70,0 %, MobileNetV2 61,6 %, EfficientNet-B0 67,4 % (≈ −11 points face à ImageNet, comme attendu).

**Alimentation** : le premier lancement a démarré sur batterie (RT-DETRv2 d'origine 48,4 ms au lieu de 47,5) ; relancé sur secteur. Le banc refuse désormais de démarrer un modèle sur batterie et note l'état à la fin de chaque modèle (`ac_power_at_end`).

**Correction 6 — précision mixte sur un état inexact.** RT-DETRv2, première passe (`runs/p3/old-rule-no-mixed-on-inexact/`) : l'INT8 transformer en 2 tuiles (40,6 ms) répare le défaut mais rate le budget de justesse (−0,29 point, borne −0,59) ; la précision mixte étant interdite sur un état inexact, l'optimiseur a pris la réparation `decoder-on-1-tile` (75,9 ms), puis un INT8 mixte : **71,0 ms (0,67×)**, 50,7 AP sur val [0, 500) (CPU 50,8, original 48,0). Désormais, la recherche de précision mixte est permise sur un état inexact quand la perte moyenne de l'INT8 est déjà dans le budget (seule la borne de confiance échoue). Relancé.
- Avec la correction 6, RT-DETRv2 en NPU seul : INT8 mixte en 2 tuiles (4 groupes en flottant, dont `(rest)`), **47,3 → 42,3 ms (1,12×)**, 50,8 AP sur val [0, 500) (CPU 50,8, original 48,0) ; résultat sur 500 images d'évaluation, archivé dans `runs/p3/eval500-agreement-and-ap/`.

**Correction 7 — l'AP décide de l'exactitude, sur 1000 images (décision du 2026-10-30).** D-FINE-N d'origine : accord 0,923 (< 0,95) et AP −0,25 point (borne −0,64 avec 10 plis de 50 images) ; l'optimiseur a payé 15 ms (26,0 → 41,2 ms, `decoder-on-1-tile`) pour réparer une perte réelle de 0,2 point, dans le budget. Désormais, avec un jeu annoté, l'exactitude se juge par la métrique seule (original, réparations, candidats exacts sur un état exact) ; l'accord ne décide que sans annotations, et sert de plancher relatif (−0,02) aux candidats exacts. Jeu de budget : val2017 [2500, 3500), 20 plis de 50 images (marge attendue ~0,25 point). Paquet 1 relancé en entier pour l'homogénéité du protocole. (Premier lancement interrompu par un bogue : le paramètre `folds` écrasé par la liste des AP par pli de la référence ; test `test_optimize_judges_exactness_by_the_metric` ajouté.)

**Paquet 1, résultats (NPU seul, protocole final ; `runs/p3/optimize_*.json`)** :

| Modèle | Avant | Après | Appliqué | AP val2017 [0, 500) : CPU / avant / après |
|---|---|---|---|---|
| RT-DETRv2-R18 | 47,4 ms (AP −2,40, borne −2,77) | **40,6 ms (1,17×)** | réparation : INT8 schéma transformer, 2 tuiles | 50,8 / 48,0 / 50,8 |
| D-FINE-N | 26,0 ms (AP −0,18, borne −0,44 : exact) | 26,0 ms (1,00×) | rien | 45,8 / 45,6 / 45,6 |
| DEIMv2-N | 24,0 ms (AP −0,44, borne −0,69) | 41,2 ms (0,58×) | réparation : 1 tuile | 46,7 / 45,7 / 46,6 |

- **RT-DETRv2** : avec 1000 images, l'INT8 transformer complet passe le test (sur 500 : refusé de justesse, puis INT8 mixte à 42,3 ms). Aucune perte sur le rapport.
- **D-FINE-N** : l'original est exact ; l'INT8 complet ne gagne que 0,3 ms pour −5,9 points (perte dans le backbone, et dans `(rest)` : +2,85 points et −0,47 ms s'il reste en flottant, le groupe le plus utile) ; aucun mélange ne tient le budget (16 groupes sur 17 en flottant : encore −2,5 points). Garder en flottant la plupart des groupes du décodeur *accélère* l'INT8 (−0,1 à −0,4 ms).
- **DEIMv2-N** : réparation coûteuse mais justifiée, l'original perdant 0,93 point sur les images du rapport. INT8 CNN −22,5 points, INT8 transformer −1,2 ; mélanges sur 1 tuile au mieux −0,56 (borne −0,98). Avec un budget d'un point, l'original serait gardé : le budget est le réglage de l'utilisateur.
- Coût : ~3 h 15 pour le paquet (RT-DETRv2 ~35 min, D-FINE-N ~1 h 45 dont la recherche de précision mixte, DEIMv2-N ~1 h).

**Exemple utilisateur** (`examples/profile_and_optimize.py`, ~1 min 40 sur le NPU, ~1 min 10 sur CPU, rien à télécharger) : faux détecteur 320 px (SiLU, depthwise, LayerNorm sur 6 400 tokens, top-300 dans le graphe), poids aléatoires, garde sur les sorties brutes. Sur le NPU : accord 1,000 ; poids aléatoires −0,1 ms (top-k dépendant des données), 50 % de zéros +0,55 ms (plus lent) ; profil : la LayerNorm coûte 41 % ; optimisation 4,28 → 2,83 ms (INT8 puis sélection sur CPU) ; recommandations SiLU → ReLU −14,5 %, LayerNorm → BatchNorm −12,3 % ; déploiement rechargé (NPU + CPU), accord 0,998. Au passage, `tiles=1/2` n'est plus proposé hors NPU.

**Correction 8 — affinage de la précision mixte.** Seul le groupe le plus utile était découpé ; quand c'est `(rest)` (indécoupable), l'affinage s'arrêtait (D-FINE-N : backbone jamais découpé). Désormais : le plus utile des groupes coûteux (plus d'un quart du gain INT8) et découpables. Sans effet attendu sur D-FINE-N (gain INT8 total de 0,3 ms).
