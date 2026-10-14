# Data Splits and Controlled Imbalance Subsets

This repository provides the dataset partitions used in the study
*Channel Attention Under Class Imbalance: A Gate- and Gradient-Level Diagnostic Study
of ECA, SE, and CBAM for Plant Disease Classification*.

It contains only dataset-related material: the five-fold train/validation/test
assignments for each dataset and the training subsets of the controlled
imbalance-ratio (IR) sweep. No images, trained models, or training code are included.

## Datasets

The images are not redistributed. Download them from the original sources:

| Dataset | Crop | Source | Classes (images) | IR |
|---|---|---|---|---|
| DiaMOS | Pear | https://zenodo.org/record/5557313 | Slug 2,025; Spot 884; Curl 54; Healthy 43 | 47.1 |
| RoCoLe | Coffee | https://data.mendeley.com/datasets/c5yvn32dzg | Healthy 791; Rust 602; Red Spider Mite 167 | 4.7 |
| BARI | Sunflower | https://data.mendeley.com/datasets/b83hmrzth8/1 | Fresh Leaf 515; Leaf scars 509; Downy mildew 470; Gray mold 398 | 1.3 |

IR is the count of the most frequent class divided by the count of the least frequent class.

For BARI, the augmented image set of the public release (1,892 images) is used.

Image paths in this repository have the form `<class>/<filename>`, relative to a
dataset folder in which the images are arranged in one subfolder per class.

## Repository contents

```
splits/
  DiaMOS_splits.csv        five-fold assignments (all images, all folds)
  RoCoLe_splits.csv
  BARI_splits.csv
ir_sweep/
  DiaMOS/ir_<IR>_fold_<k>_train.csv   training images at each IR level and fold
  RoCoLe/ir_<IR>_fold_<k>_train.csv
  DiaMOS_ir_summary.csv    per-class counts before and after subsampling
  RoCoLe_ir_summary.csv
ir_dataset.py              IR subsampling procedure
make_ir_subsets.py         regenerates the files in ir_sweep/ from splits/
```

### `splits/<dataset>_splits.csv`

| Column | Meaning |
|---|---|
| `index` | image index within the dataset |
| `image` | relative image path, `<class>/<filename>` |
| `class`, `label` | class name and integer label |
| `fold` | fold number (1–5) |
| `split` | `train`, `val`, or `test` for that fold |

Each dataset is divided into five stratified folds. Within each fold, 64% of the
images are used for training, 16% for validation (model selection and early
stopping), and 20% for testing. Every image appears in the test set of exactly one
fold, and no image is shared between the training, validation, and test sets of the
same fold.

### `ir_sweep/`

The controlled sweep uses IR ∈ {1, 2, 5, 10, 20, 40} on DiaMOS and RoCoLe with
seed 42. Only the training portion of each fold is subsampled; validation and test
sets are the unchanged fold splits at every IR level. No synthetic upsampling is applied.

- **When the target IR does not exceed the natural training IR** (`subsample_majority`):
  the minority class is kept intact and every other class is capped at
  `minority count × target IR` images. This applies to all DiaMOS levels and to
  RoCoLe at IR = 1 and 2.
- **When the target IR exceeds the natural training IR** (`subsample_minority`):
  the minority class is reduced to `largest class count / target IR` images, and
  the other classes keep their natural counts. This applies to RoCoLe at IR ≥ 5.

The minority class is Healthy for DiaMOS and Red Spider Mite for RoCoLe.

## Regenerating the IR subsets

Requires Python 3 and NumPy.

```
python make_ir_subsets.py --dataset DiaMOS --out ir_sweep_regenerated
python make_ir_subsets.py --dataset RoCoLe --out ir_sweep_regenerated
```

The generated lists are identical to those in `ir_sweep/`.

## License

The split files and scripts in this repository are released under the MIT License
(see `LICENSE`). The images remain subject to the licences of the original datasets.

## Citation

Citation details will be added upon publication.
