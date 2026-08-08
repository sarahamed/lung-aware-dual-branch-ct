# Data

This directory does not contain the CT scan images themselves — the dataset
(IQ-OTHNCCD lung cancer CT scan dataset) is third-party and distributed
separately.

## Obtaining the dataset

The pipeline downloads the dataset automatically via
[`kagglehub`](https://pypi.org/project/kagglehub/) the first time
`src/preprocessing.py`'s `find_data_root` / dataset-loading code runs — no
manual download is required. This requires a Kaggle account and API
credentials configured per kagglehub's instructions.

If you prefer to download manually, get the dataset from:

- Kaggle: "The IQ-OTHNCCD lung cancer dataset"
  https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset

## Expected directory layout

After download, the pipeline expects three class subfolders (folder names on
disk, per `config.CLASS_FOLDERS`):

```
<DATA_ROOT>/
├── Bengin cases/
├── Malignant cases/
└── Normal cases/
```

## Citation

If you use this dataset, please cite the original dataset authors per the
Kaggle dataset page's citation guidance.
