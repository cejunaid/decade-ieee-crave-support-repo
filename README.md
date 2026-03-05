# Multimodal Misinformation Evaluation Resources

This repository contains resources for evaluating multimodal misinformation detection models. It includes **two experiment directories**:

* `RedDot`
* `outcontext-misinfo-progress`

Each directory contains **evaluation-ready datasets, pretrained model checkpoints, and evaluation scripts** for reproducing the reported results.

---

# Repository Structure

```
RedDot/
│
├── checkpoints/
│   └── best model checkpoint (.pt)
│
├── data/
│   └── eval/
│       ├── DP/
│       ├── FIVEPILS/
│       ├── MMFAKEBENCH/
│       └── VERITE/
│
└── checkpoint-evaluation.py


outcontext-misinfo-progress/
│
├── checkpoints/
│   └── best model checkpoint (.pt)
│
├── data/
│   └── eval/
│       ├── DP/
│       ├── FIVEPILS/
│       ├── MMFAKEBENCH/
│       └── VERITE/
│
└── checkpoint-evaluation.py
```

---

# Evaluation Datasets

Each experiment directory contains a `data/eval/` folder with **pre-computed feature files** for evaluation on the following datasets:

* **DP**
* **FIVEPILS**
* **MMFakeBench**
* **VERITE**

These files contain **pre-extracted multimodal features** (e.g., image embeddings and associated metadata) so that evaluation can be run directly **without recomputing features**.

---

# Model Checkpoints

The `checkpoints/` directory in each experiment folder contains the **best-performing trained model checkpoint** used for evaluation.

These checkpoints can be loaded directly by the evaluation script.

---

# Running Evaluation

To evaluate a model checkpoint on the datasets:

```bash
python checkpoint-evaluation.py
```

The script loads:

* the model checkpoint from `checkpoints/`
* the evaluation features from `data/eval/`

and reports the model’s performance on the available datasets.

---

# Notes

* All datasets in `data/eval/` are **evaluation feature files only**, not the original raw datasets.
* Feature files are provided in `.npy` format to allow **fast reproducible evaluation**.
* The provided checkpoints correspond to the **best validation performance** during training.

---
