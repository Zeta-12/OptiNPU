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

## Suivi

- [x] L0 — 2026-10-27. OpenVINO 2026.4.1, PyTorch 2.14.1+xpu, NNCF 3.4. ResNet-50 FP16 : CPU 31,5 ms, GPU 4,3 ms, NPU 4,1 ms (287 fps).
- [x] L1 — 2026-10-27. `Detector`, export, runtime, CLI, `diagnose`, 19 tests. D-FINE-N/S correctes sur NPU ; RT-DETRv2 dégradé sur NPU (correct sur GPU FP16).
- [x] L2 — 2026-10-27. mAP COCO (sous-ensemble val2017), INT8 NNCF, profilage par couche et par étage, CLI `export | eval | profile`, 22 tests. Conclusion : l'INT8 n'accélère pas D-FINE sur NPU (voir journal).
- [x] L3 — 2026-10-27. `Detector.train()` / `optinpu train` sur dataset au format COCO, iGPU (XPU), EMA, tête réinitialisée si nouvelles classes, `from_checkpoint`. D-FINE-N fine-tuné sur COCO-animaux (10 classes) : 62,6 AP (PyTorch), 63,0 AP sur NPU, 26 ms. 26 tests.
- [x] L4 — 2026-10-27. `npu_lint` (CLI `lint`), `diagnose.locate_divergence_split`, exécution hybride `npu+gpu` (la plus rapide sur les 3 modèles), RT-DETRv2 exact sur NPU (`NPU_TILES=1`), packaging v0.1.0 (wheel, licence Apache-2.0), dépôt git local, 32 tests. QAT reportée à M2 : sans intérêt tant que le modèle ne profite pas de l'INT8 sur ce NPU.
- [x] L5 — 2026-10-28. Note de faisabilité ci-dessous. Livrable : `train(..., backbone_device="npu")` (backbone gelé sur NPU, ×2 sur le fine-tuning). Calcul de gradients sur NPU abandonné. Une seule tuile NPU devient le réglage par défaut des modèles DETR. Nettoyage final de la librairie (ruff, code mort retiré), 35 tests.

- [x] M2 — 2026-10-28. `npu_lint` fondé sur les détections, architecture OptiDet et perte d'entraînement (`optidet-n`, backbone RepVGG-A0 ImageNet), balayage de latence de 36 candidats, premier entraînement (10 000 images COCO, 10 epochs : 8,5 AP), distillation depuis D-FINE-S (10,0 AP, +18 %), QAT (INT8 sur NPU 10,0 AP, aucune perte, 3,37 ms, exact en 2 tuiles), 53 tests. L'AP atteignable reste à mesurer en M3 (COCO complet).
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
