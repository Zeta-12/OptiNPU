# OptiNPU — Plan du projet

## Contexte matériel

- CPU : Intel Core Ultra 5 235U
- NPU : Intel AI Boost (~11-13 TOPS INT8)
- iGPU : Intel Graphics
- RAM : 15,5 Go
- Python : environnement virtuel 3.12 (3.14 trop récent pour OpenVINO / PyTorch XPU / NNCF)

## Contraintes structurantes

1. **Le NPU Intel ne fait que de l'inférence** (OpenVINO, plugin NPU). L'entraînement se fait sur iGPU (PyTorch XPU) ou CPU, puis le modèle est exporté et exécuté sur le NPU. L'entraînement sur NPU reste une piste exploratoire (L5).
2. **Pas de SOTA COCO absolu depuis un portable.** L'objectif retenu : **le meilleur AP COCO à latence donnée sur NPU edge** (frontière de Pareto AP / latence NPU INT8, batch 1). Les entraînements COCO complets nécessiteront du GPU cloud.
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

### État de l'art à exploiter (à revérifier au lancement)

- **DETR temps réel** : RT-DETRv2/v3, D-FINE (raffinement fin par distributions), DEIM / DEIMv2 (appariement un-à-un densifié, DINOv3), RF-DETR / LW-DETR (backbone DINOv2, NAS).
- **YOLO** : YOLOv10, YOLO26 (sans NMS, orientés edge), YOLOv12 (attention par zones).
- **Backbones auto-supervisés** : DINOv2 / DINOv3 comme enseignants.

### Thèse : un détecteur conçu pour le NPU

1. **Sans NMS** (appariement un-à-un) : tout le graphe sur NPU.
2. **Sans attention déformable** (grid_sample mal supporté) : attention par fenêtres/zones à formes statiques ou alternatives convolutionnelles.
3. **Conçu pour l'INT8** : activations bornées, convolutions reparamétrables (RepVGG), normalisations fusionnables, QAT intégrée.
4. **Distillation** depuis DINOv3 ou D-FINE-X.
5. **NAS guidée par la latence NPU mesurée** via `optinpu.bench` (pas les FLOPs).
6. Têtes de régression inspirées de D-FINE / DEIM.

### Jalons

| Phase | Contenu |
|---|---|
| M0 | Baselines (D-FINE-S/N, DEIM-S, RF-DETR-N, YOLO26-n) mesurées sur le NPU : AP, latence INT8/FP16 → frontière de Pareto de référence |
| M1 | Ablations rapides sur ~10 % de COCO, 12 époques |
| M2 | NAS guidée par latence NPU, distillation, QAT |
| M3 | Entraînement COCO complet (GPU cloud), comparaison AP-latence |
| M4 | Publication : poids, rapport technique, éventuel article |

**Métrique cible** : dominer la frontière (AP COCO val, latence NPU INT8 batch 1) face à D-FINE, DEIM et YOLO26 à taille comparable.

---

## Suivi

- [x] L0 — 2026-10-27. OpenVINO 2026.4.1, PyTorch 2.14.1+xpu, NNCF 3.4. ResNet-50 FP16 : CPU 31,5 ms, GPU 4,3 ms, NPU 4,1 ms (287 fps).
- [x] L1 — 2026-10-27. `Detector`, export, runtime, CLI, `diagnose`, 19 tests. D-FINE-N/S correctes sur NPU ; RT-DETRv2 dégradé sur NPU (correct sur GPU FP16).
- [x] L2 — 2026-10-27. mAP COCO (sous-ensemble val2017), INT8 NNCF, profilage par couche et par étage, CLI `export | eval | profile`, 22 tests. Conclusion : l'INT8 n'accélère pas D-FINE sur NPU (voir journal).
- [x] L3 — 2026-10-27. `Detector.train()` / `optinpu train` sur dataset au format COCO, iGPU (XPU), EMA, tête réinitialisée si nouvelles classes, `from_checkpoint`. D-FINE-N fine-tuné sur COCO-animaux (10 classes) : 62,6 AP (PyTorch), 63,0 AP sur NPU, 26 ms. 26 tests.
- [x] L4 — 2026-10-27. `npu_lint` (CLI `lint`), `diagnose.locate_divergence_split`, exécution hybride `npu+gpu` (la plus rapide sur les 3 modèles), RT-DETRv2 exact sur NPU (`NPU_TILES=1`), packaging v0.1.0 (wheel, licence Apache-2.0), dépôt git local, 32 tests. QAT reportée à M2 : sans intérêt tant que le modèle ne profite pas de l'INT8 sur ce NPU.
- [x] L5 — 2026-10-28. Note de faisabilité ci-dessous. Livrable : `train(..., backbone_device="npu")` (backbone gelé sur NPU, ×2 sur le fine-tuning). Calcul de gradients sur NPU abandonné. Une seule tuile NPU devient le réglage par défaut des modèles DETR. Nettoyage final de la librairie (ruff, code mort retiré), 35 tests.

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

Enseignements pour OptiDet (M0-M2) :
- **Le placement hybride est la référence à battre** : OptiDet doit être plus rapide en NPU seul que D-FINE en NPU+GPU, ou être conçu lui-même pour un placement hybride (tête NPU lourde, décodeur GPU minimal).
- **D-FINE n'est pas limité par le calcul sur le NPU** mais par le nombre de petites ops et les faibles nombres de canaux. Le backbone (16 à 32 canaux à 320×320) et le décodeur (300 requêtes, `integral` 300×4×33, sommes de l'attention déformable) sont les points chauds. L'INT8 ajoute des quantize/dequantize sans réduire ce surcoût.
- À l'inverse, **un CNN large et régulier (ResNet-50) gagne ×2,4 en INT8.** OptiDet doit viser des ops peu nombreuses, larges et alignées (canaux multiples de 32 ou 64), avec un décodeur compact, pour exploiter l'INT8.
- Sur ce portable, le CPU en INT8 (49 ms) rivalise avec le NPU en FP16 pour D-FINE-S : le NPU n'est intéressant que pour des modèles adaptés à lui. C'est la justification même d'OptiDet.
- **Le raffinement des boîtes par distributions (D-FINE FDR) résiste au bruit FP16** du NPU (erreur de 0,025 sur les boîtes), contrairement à la régression directe de RT-DETRv2 (0,17). C'est un argument pour une tête de type D-FINE.
- Tout ce qui ne dépend que de la taille d'entrée doit être une constante au moment de l'export.
