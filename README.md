# Lung-Aware Dual-Branch CT Classification with Cross-Attention

Code accompanying the *Scientific Reports* manuscript on a dual-branch
(global + lung-ROI local) EfficientNetV2-S architecture with bidirectional
cross-attention fusion for 3-class lung cancer classification (Benign /
Malignant / Normal) on CT scans, including the full explainability (XAI),
robustness, and external-validation analysis performed during peer review.

## Summary

The model takes two views of the same CT slice as input: the full image
("global" branch) and an automatically segmented lung-region crop ("local"
branch), each encoded by a shared EfficientNetV2-S backbone. The two branches
exchange information through bidirectional multi-head cross-attention before
fusion and classification.

This repository contains a single end-to-end notebook
(`notebooks/complete_pipeline.ipynb`)
covering training, explainability (Grad-CAM, Grad-CAM++, SmoothGrad,
Integrated Gradients, Occlusion Sensitivity), and the full peer-review
revision pipeline: a dataset-wide segmentation audit, segmentation/crop-window
perturbation sensitivity, external-dataset validation, image-corruption
robustness testing, expanded XAI statistics, and multi-seed stability
analysis. The same logic is also available as importable modules under `src/`
for anyone who wants to reuse individual pieces in their own code.

## Repository structure

```
lung-aware-dual-branch-ct/
├── README.md
├── LICENSE
├── requirements.txt
├── config.py                  # central, env-var-driven configuration
├── notebooks/
│   └── complete_pipeline.ipynb  # the full pipeline, end to end
├── data/
│   └── README.md              # how to obtain the dataset
├── src/
│   ├── preprocessing.py       # lung-ROI segmentation, cropping, dataset builders
│   ├── model.py                # dual-branch cross-attention architecture
│   ├── train.py                # training protocols (primary + multi-seed)
│   ├── evaluate.py             # checkpoint evaluation, metrics, plots
│   ├── xai.py                  # Grad-CAM / Grad-CAM++ / SmoothGrad / IG / Occlusion
│   ├── segmentation_audit.py   # dataset-wide lung-mask coverage audit
│   ├── robustness.py           # image-corruption + segmentation robustness
│   └── external_validation.py  # external-dataset class mapping and scoring
└── results/
    └── README.md               # what a pipeline run produces, where
```

## Installation

```bash
git clone <this-repository-url>
cd lung-aware-dual-branch-ct
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU for training in reasonable time
(inference and small-scale analysis run on CPU as well, just slower).

## Configuration

All paths are controlled via environment variables (see `config.py`) so the
code runs unmodified on any machine:

```bash
export OUTPUT_ROOT=./ct_lungcancer_experiments   # where run folders are written
export RUN_NAME=my_run_20260101_120000            # which run folder to use/create
```

If you're reproducing the manuscript's reported results rather than training
from scratch, leave `RUN_NAME` unset — it defaults to the run identifier used
for the reported checkpoint; see `data/README.md` for how to obtain that
checkpoint.

## Dataset

This code uses the IQ-OTHNCCD lung cancer CT scan dataset (downloaded
automatically via `kagglehub`; requires Kaggle API credentials). See
`data/README.md` for details and citation. The external-validation step
(`src/external_validation.py`) additionally downloads a second, independent
chest-CT dataset for out-of-distribution testing.

## Running the pipeline

Open and run `notebooks/complete_pipeline.ipynb` top to bottom. It covers, in order:

1. **Setup** — portable, environment-variable-driven configuration.
2. **Shared preprocessing / XAI utilities** — lung-ROI segmentation, cropping,
   Grad-CAM/Grad-CAM++/SmoothGrad/Integrated Gradients/Occlusion.
3. **Model, data, training, checkpoint verification** — training is included
   but skipped by default (`TRAIN_FROM_SCRATCH = False`); by default the
   notebook loads the existing, manuscript-verified checkpoint and verifies
   its reported test accuracy before continuing. Set that flag to `True` to
   train a new model from scratch instead.
4. **Tasks 1-6** — the full peer-review revision pipeline: segmentation
   audit, segmentation/crop-window sensitivity, external validation,
   perturbation robustness, expanded XAI statistics (N=108,
   Friedman + Holm-Bonferroni-corrected Wilcoxon), and multi-seed stability
   analysis.

The `src/` modules mirror the same logic as importable functions, for anyone
who wants to reuse individual pieces (e.g. just the preprocessing, or just
the XAI methods) outside the notebook.

## Citation

If you use this code, please cite:

> Hamed, S. "Lung-Aware Dual-Branch Cross-Attention Network for Explainable Lung Cancer Classification from CT Images." *Scientific Reports* (2026).
> DOI: [MANUSCRIPT DOI PLACEHOLDER]

For the code itself, please also cite this repository's archived release:

> Hamed, S. "lung-aware-dual-branch-ct" (v1.0.0). Zenodo.
> DOI: [ZENODO DOI PLACEHOLDER — see CITATION.cff]

Machine-readable citation metadata is provided in `CITATION.cff`.

If you use the IQ-OTHNCCD dataset, please also cite it per the guidance in
`data/README.md`.

## License

MIT — see `LICENSE`.
