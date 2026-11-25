# The Static Validation Fallacy in Active Learning Research

Code for the paper *The Static Validation Fallacy in Active Learning Research* by Moussa Kassem Sbeyti\*, Dani Manjah\*, Christophe De Vleeschouwer and Nadja Klein (\*equal contribution; Karlsruhe Institute of Technology and UCLouvain).

Active learning (AL) methods are usually compared on one validation set. The paper shows that the resulting ranking does not transfer to other subsets, datasets or cameras, measures the cost with *deployment regret*, and proposes to rank AL methods on the mid-uncertainty tertile of the validation set instead of the full set.

The supplementary material of the paper is [`supplement.pdf`](supplement.pdf).

## What is in this repository

| Study | Setting | Status |
|---|---|---|
| I | Pool-based: ten AL methods trained on KITTI with EfficientDet-D0, evaluated on KITTI and cross-domain on CODA and BDD100K | This package (`static_val`) |
| II | Stream-based: six admission AL methods with a YOLO11n student on the 15 WALT cameras | Being added. Training code and data: [CSBAD](https://github.com/manjahdani/CSBAD) |

Training the detectors and running the pool-based AL methods is done with [uncertainty-detection-autolabeling](https://github.com/mos-ks/uncertainty-detection-autolabeling) (section *Active Learning*). That repository also provides the TFRecord creators for KITTI and CODA and the validation-set inference. This repository starts from the ten trained detectors and writes its analyses as CSV tables.

## Outputs

`static-val analyze` writes these tables to `results/analysis`. The repository contains no data, trained models or results; every table is computed from your own runs.

| File | Content |
|---|---|
| `stage1_subset_ranks.csv` | Rank of each AL method over random subsets of each validation set |
| `stage1_cross_dataset_rho.csv` | Spearman rho between the KITTI, CODA and BDD rankings |
| `stage2_transferability.csv` | Spearman rho between each KITTI stratum ranking (class, depth tertile, entropy tertile) and the CODA and BDD rankings |
| `stage3_stratified_map.csv` | In-domain mAP of every method on the full set and on each uncertainty tertile |
| `stage3_deployment_regret.csv` | Deployment regret when the AL method is selected on the full set or on an uncertainty tertile, for entropy and for 1 - max class score |
| `stage3_protocol_ranking.csv` | Rank under aggregate KITTI mAP and under the protocol, against the cross-dataset score |

## Setup

```bash
uv sync
```

Detector evaluation and the BDD TFRecord run in the environment of uncertainty-detection-autolabeling (TensorFlow). `static-val evaluate --python <that interpreter>` puts this package on its path, so nothing else needs to be installed there.

## Configuration

`configs/kitti.yaml` holds all paths, the AL methods and the run name of each trained detector, the subset settings (5 subsets of 20 %, seed 42) and the class groups. Relative paths resolve against the config file. `--data-root` and `--autolabeling-dir` override the two roots.

| Key | Content |
|---|---|
| `methods` | AL method name and the run name of its detector |
| `detector.checkpoint` | Checkpoint directory per run |
| `detector.predictions` | `prediction_data_val.txt` per run: one dict per detection with `image_name`, `class` (1-based), `bbox` as `[y1, x1, y2, x2]`, `det_score`, `entropy` |
| `kitti.*` | KITTI labels, images, validation index list, validation TFRecord, depth maps |
| `targets.*` | CODA and BDD100K validation TFRecords, image counts and first source id |

## Run Study I

| Step | Command | Result |
|---|---|---|
| 1 | Train the ten AL methods and run validation inference (`inspector.py --mode 6`) per model in uncertainty-detection-autolabeling | Checkpoints and prediction files |
| 2 | `static-val bdd-tfrecord` | BDD100K val TFRecord with the classes shared with KITTI (`rider` becomes `cyclist`) |
| 3 | Generate the KITTI validation depth maps with [DepthPrior](https://github.com/mos-ks/DepthPrior) (section *Dataset Preparation*) and place them in `kitti.depth_dir` | One `<image>_depth.npy` per validation image |
| 4 | `static-val strata` | Image-id lists of all subsets and tertiles in `results/strata` |
| 5 | `static-val evaluate --python <detector env python>` | COCO metrics per method and subset in `results/detector_eval` (resumable) |
| 6 | `static-val match` | Per-detection true-positive tables in `results/cache` |
| 7 | `static-val analyze` | CSV tables in `results/analysis` |

Add `--config <file>` to any command to use another configuration.

## Method details

- **AL methods.** Random, an entropy grid (Ent, Cal-Ent, CB-Ent, Cal-CB-Ent: calibration and class-balancing), box uncertainty (EpUnc, AlBox, Cal-AlBox, Cal-EpBox) and Ent+Al. Each method is one trained detector.
- **Class groups.** `head`: Car, Van, Truck, Pedestrian, Cyclist. `tail`: Person_sitting, Tram. `shared` (present in KITTI, CODA and BDD100K): Car, Truck, Pedestrian, Cyclist.
- **Cross-domain ranking.** Each target ranks the AL methods by their mean mAP over five random 20 % subsets of the target validation set.
- **Transferability.** Spearman rho between the ranking on a KITTI stratum and the ranking on each target. Tied values get their average rank. Depth and entropy strata are scored on head-class mAP (`--stage2-classes`).
- **Entropy tertiles for the diagnosis.** Uncertainty is computed per detection and averaged per image. Each method's validation images are sorted by their mean detection entropy and split into three equal parts, so every detector is evaluated on its own tertiles.
- **Uncertainty tertiles for selection.** One partition shared by all methods, so every method is ranked on the same images. An image's score is the mean over methods of that method's mean per-image uncertainty, either entropy or 1 - max class score. Images without detections are left out. mAP is recomputed from the cached true-positive tables with COCO matching at IoU 0.50:0.05:0.95 and 101-point interpolation, on all classes (`--stage3-classes`).
- **Deployment regret.** Deploying the method that is best on the source slice gives regret = deployed mAP minus best mAP on each target, in mAP points (at most 0). Recovery is the share of the full-set rule's summed regret that a rule removes.
- **Protocol ranking.** The protocol score of a method is its shared-class mAP on its own mid-entropy tertile. The cross-dataset score is the mean of the method's min-max normalised mAP on KITTI, CODA and BDD100K. `rho_aggregate` and `rho_protocol` are the Spearman correlations of the aggregate KITTI mAP and of the protocol score with that cross-dataset score.
- **Depth tertiles.** Per image, the mean over labelled objects of the mean depth-map value inside each box, using the Depth Anything maps from DepthPrior: the smallest third is `near`, the largest `far`. For maps that hold inverse depth (larger is closer), use `static-val strata --inverse-depth`.

## Tests

```bash
uv run ruff check . && uv run pytest
```
