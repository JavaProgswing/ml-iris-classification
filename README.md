# Iris Species Classification

A beginner multiclass-classification project. It predicts which of three iris
species a flower belongs to from four measurements, using logistic regression.

This project is educational only.

## Dataset

Download Kaggle's [Iris dataset](https://www.kaggle.com/datasets/uciml/iris) and
place `Iris.csv` at:

```text
data/raw/Iris.csv
```

With the Kaggle CLI:

```bash
kaggle datasets download -d uciml/iris -p data/raw --unzip
```

The file has 150 rows and is perfectly balanced (50 of each species). The `Id`
column is dropped; the four measurement columns are the features.

## Workflow

1. Load the raw CSV and remove exact duplicate rows.
2. Keep the four measurement columns; drop `Id`.
3. Make a stratified 80/20 train/test split.
4. Standardize the measurements.
5. Train logistic regression in a leakage-safe pipeline.
6. Compare against a majority-class dummy model.
7. Evaluate with accuracy and macro precision/recall/F1, a confusion matrix, and
   five-fold stratified cross-validation.

## Results

| Measurement | Result |
|---|---:|
| Dummy test accuracy | 33.3% |
| Logistic regression test accuracy | 93.3% |
| CV mean accuracy | 95.8% |

The species are almost linearly separable, so a simple model does very well. The
few errors are between versicolor and virginica, which overlap in petal size.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

The confusion matrix is written to `reports/figures/confusion_matrix.png`.

## Layout

```text
data/raw/         downloaded dataset (git-ignored - see Dataset above)
src/preprocess.py loading, cleaning, scaler
src/train.py      pipeline, training loop, evaluation, figure
src/evaluate.py   classification metric helpers
main.py           entry point
```

## Why this project

Inspired by classmate iris projects (e.g. Whimsical-Maverick's
`Flower-classifier-using-iris-dataset`, adityaxdubey's `Iris-Flower-Detection`).
It adds a multiclass example to the sibling `ml-*` folders.
