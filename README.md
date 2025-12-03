# UCI HAR – Human Activity Recognition Using Smartphones

## Overview

This repository presents an end-to-end Human Activity Recognition (HAR) pipeline developed in MATLAB using the UCI HAR dataset. The objective is to classify six human activities based on inertial sensor signals captured from a smartphone worn at the waist:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

The workflow integrates:

- Loading and preprocessing of raw and pre-computed data
- Signal-processing-based feature extraction
- Statistical feature selection
- Training and evaluation of machine learning models
- Visualization utilities

The repository includes a MATLAB App Designer GUI for interactive demonstrations (`create_app_clean.m`). When available, MATLAB toolboxes (Statistics and Machine Learning Toolbox, Signal Processing Toolbox) enhance functionality; toolbox-free fallbacks are provided to ensure full operation on minimal MATLAB installations.

---

## Dataset Description

- **Source:** UCI Machine Learning Repository — Human Activity Recognition Using Smartphones Dataset
- **Device:** Samsung Galaxy S smartphone worn on the waist
- **Sampling Rate:** 50 Hz

### Sensors

- 3-axis accelerometer (body + total acceleration)
- 3-axis gyroscope

### Windowing

````markdown
# UCI HAR – Human Activity Recognition Using Smartphones

## Overview

This repository implements an end-to-end Human Activity Recognition (HAR) pipeline in MATLAB using the UCI HAR dataset. The goal is to classify six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) from smartphone inertial sensors using signal processing, feature engineering, and classic machine learning (k‑NN baseline).

Toolbox-free fallbacks are provided so scripts run on minimal MATLAB installations; optional MATLAB toolboxes (Statistics and Machine Learning Toolbox, Signal Processing Toolbox) enhance some functionality.

---

## Dataset

- Source: UCI Machine Learning Repository — "Human Activity Recognition Using Smartphones Dataset"
- Device: Smartphone worn at the waist (accelerometer + gyroscope)
- Sampling: 50 Hz, windowed into 2.56 s frames (128 samples)

The dataset is included in this repository under `UCI HAR Dataset/`. Do not move this folder — code expects the dataset at `fullfile(pwd,'UCI HAR Dataset')` relative to the project folder.

---

## Project Layout

```
UCI HAR Dataset/
results_figs/
src/ (scripts and helpers)
  load_prepare.m
  extract_features.m
  feature_selection.m
  train_models.m
  tune_k_values.m
  evaluate_models.m
  generate_confusion_matrices.m
  pipeline_run.m
  create_app_clean.m
  eda.m
  spectrogram_utils.m
  regenerate_results.m
  predict_knn_model.m
  save_trained_knn.m
  show_results.m
har_models_results.mat
knn_model.mat
```

Key points:
- Run the full pipeline with `pipeline_run.m`.
- Recreate presentation figures with `regenerate_results.m` (writes to `results_figs/`).
- Launch the interactive demo with `create_app_clean.m`.

---

## Key Scripts (one-line summary)

- `create_app_clean.m` — MATLAB App Designer GUI for interactive demos.
- `load_prepare.m` — load feature matrices / raw signals and perform consistency checks.
- `extract_features.m` — extract features from raw Inertial Signals (toolbox-free).
- `feature_selection.m` — ReliefF / F‑score / PCA / sequential selection.
- `train_models.m` — train and validate k‑NN and nearest-centroid baselines.
- `tune_k_values.m` — optional k‑NN hyperparameter tuning helper.
- `evaluate_models.m` — metrics and confusion matrices.
- `regenerate_results.m` — regenerate presentation-quality figures in `results_figs/`.

---

## Figures

All generated figures are stored in `results_figs/` and referenced by the README via relative paths:

![Class distribution](results_figs/class_distribution.png)

![Sample time series](results_figs/sample_time_series.png)

![Sample spectrogram](results_figs/sample_spectrogram.png)

![kNN confusion matrix](results_figs/knn_confusion_matrix.png)

---

## Quick Start

1. Launch GUI:

```matlab
app = create_app_clean();
```

2. Extract features (example):

```matlab
[F, L] = extract_features('UCI HAR Dataset');
```

3. Optional feature selection and tuning:

```matlab
optsFS.numFeatures = 50;
[selIdx, selNames, Fsel] = feature_selection(F, L, 'relieff', optsFS);
[bestK, kStats] = tune_k_values(Fsel, L);
```

4. Train models (example):

```matlab
opts = struct('bestK', bestK, 'K', 5, 'tuneK', false);
results = train_models(Fsel, L, opts);
```

5. Full pipeline and reproduce figures:

```matlab
pipeline_run('UCI HAR Dataset', struct('saveResults', true));
regenerate_results;
```

---

## Output Files

- `results_figs/` — generated visualizations
- `har_models_results.mat` — evaluation summaries
- `knn_model.mat` — saved k‑NN model

---

## Notes

- The dataset is intentionally included for reproducibility; large files may trigger GitHub size warnings. If you prefer an external download, I can add a script or switch to Git LFS on request.
- If you want a shorter academic-style README, tell me which sections to trim.

---

If you'd like further edits (different order, shorter or more academic tone, or translation to another language), tell me which headings to keep or remove.
````
    3. **Feature selection** (optional)
    4. **Training & validation** of ML models
    5. **Evaluation & visualization** (confusion matrices, etc.)

- **`regenerate_results.m`**
  - Lightweight script that only regenerates the key **presentation figures** in `results_figs`:
    - `class_distribution.png`
    - `sample_time_series.png`
    - `sample_spectrogram.png` (reference‑style spectrogram for sample 3676)
    - `knn_confusion_matrix.png`

---

## Figures

All figures generated by the project are stored in:

`human+activity+recognition+using+smartphones/results_figs/`

![Class distribution](results_figs/class_distribution.png)

![Sample time series](results_figs/sample_time_series.png)

![Sample spectrogram](results_figs/sample_spectrogram.png)

![kNN confusion matrix](results_figs/knn_confusion_matrix.png)

These figures can be reproduced at any time using the scripts mentioned above.

---

## Quick Start

### Launch the GUI

```matlab
app = create_app_clean();
```

### Extract Features

```matlab
[F, L] = extract_features('UCI HAR Dataset');
```

### Feature Selection

```matlab
optsFS.numFeatures = 50;
[selIdx, selNames, Fsel] = feature_selection(F, L, 'relieff', optsFS);
```

### Optional: Hyperparameter Tuning

```matlab
[bestK, kStats] = tune_k_values(Fsel, L);
```

### Model Training

```matlab
opts = struct('bestK', bestK, 'K', 5, 'tuneK', false);
results = train_models(Fsel, L, opts);
```

### Full Pipeline

```matlab
pipeline_run('UCI HAR Dataset', struct('saveResults', true, 'summaryOnly', false));
```

### Recreate Presentation Figures

```matlab
cd human+activity+recognition+using+smartphones
regenerate_results;
```

### Generate Confusion Matrices

```matlab
generate_confusion_matrices('UCI HAR Dataset');
```

---

## Conclusion

This repository delivers a complete and reproducible **HAR analysis framework** in MATLAB.
The workflow integrates raw-signal processing, feature engineering, traditional ML classification, cross-validated evaluation, and fully automated visualization.
The inclusion of a MATLAB GUI makes the project suitable for:

* Educational use
* Research comparisons
* Live demonstrations
* Rapid prototyping

Toolbox-free alternatives allow the pipeline to operate on a wide range of MATLAB installations while maintaining high interpretability and methodological clarity.

---

## Output Files

* **results_figs/** — visualizations (EDA, spectrograms, confusion matrices)
* **har_models_results.mat** — stored model metrics
* **knn_model.mat** — trained k-NN classifier


