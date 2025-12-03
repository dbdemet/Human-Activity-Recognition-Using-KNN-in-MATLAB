## UCI HAR – Human Activity Recognition Using Smartphones

### Project Overview

This repository contains an end‑to‑end **human activity recognition (HAR)** pipeline in MATLAB, built on top of the well‑known **UCI HAR** dataset. The goal is to automatically classify 6 different human activities from smartphone inertial sensor signals (3‑axis accelerometer and gyroscope) mounted on the waist:

- **WALKING**
- **WALKING_UPSTAIRS**
- **WALKING_DOWNSTAIRS**
- **SITTING**
- **STANDING**
- **LAYING**

The project demonstrates a complete workflow:

- **Loading and preprocessing of raw data**
- **Feature extraction (using signal processing techniques)**
- **Feature selection (using statistical methods)**
- **Training and validation of machine learning models**
- **Visualization of results and GUI presentation**

Alongside the command‑line pipeline, a **MATLAB App Designer GUI** (`create_app_clean.m`) is provided for interactive exploration and real‑time demos.

**MATLAB Toolboxes:** The project makes use of **Statistics and Machine Learning Toolbox** and **Signal Processing Toolbox** when available, but also includes **fallback, toolbox‑free implementations**, so the core pipeline remains usable even in more limited MATLAB installations.

---

## Dataset

- **Source**: UCI Machine Learning Repository – *“Human Activity Recognition Using Smartphones Dataset”*  
- **Device**: Samsung Galaxy S smartphone worn on the waist  
- **Sensors**:
  - 3‑axis **accelerometer** (total acceleration and body acceleration – `body_acc`)
  - 3‑axis **gyroscope** (angular velocity)
- **Sampling frequency**: \( f_s = 50 \,\text{Hz} \)
- **Raw windowing**:
  - 2.56 s windows (128 samples) for each subject and experiment
  - Each window contains 9 time‑series signals:  
    `body_acc_x/y/z`, `body_gyro_x/y/z`, `total_acc_x/y/z`
- **Pre‑computed features**:
  - `X_train.txt`, `X_test.txt` contain ≈560 features per window
  - `y_train.txt`, `y_test.txt` contain the 6 activity labels

In this project:

- Two data sources are used:
  - The **pre‑computed feature matrices** (X, y) for quick training and baselines
  - The **raw “Inertial Signals”** folders for custom feature extraction and spectrogram‑based analysis

---

## Project Structure (Workspace overview)

- **`UCI HAR Dataset/`** – Original dataset (train/test splits and Inertial Signals)
- **Main MATLAB scripts**:
  - `load_prepare.m`, `extract_features.m`, `feature_selection.m`, `train_models.m`, `tune_k_values.m`, `evaluate_models.m`, `generate_confusion_matrices.m`, `pipeline_run.m`, `create_app_clean.m`, `eda.m`, `spectrogram_utils.m`, `regenerate_results.m`, `predict_knn_model.m`, `save_trained_knn.m`, `show_results.m`
- **Models and results**:
  - `har_models_results.mat` – trained model metrics
  - `knn_model.mat` – saved k‑NN model
- **Figures**:
  - `human+activity+recognition+using+smartphones/results_figs/` – EDA plots, spectrograms, confusion matrices, sample time‑series, etc.

With this layout you can:

- Run a full training & evaluation pipeline with a single command (`pipeline_run`)
- Re‑generate all presentation figures (`regenerate_results`)
- Launch the GUI (`create_app_clean`) for interactive demos

---

## Key Scripts and Responsibilities

### 🎯 GUI Demo
- **`create_app_clean.m`** – MATLAB App Designer GUI
  - Loads the dataset and/or pre‑computed feature set
  - Shows raw signals and normalized features
  - Performs real‑time activity prediction with the trained k‑NN model
  - Designed for presentations and interactive demos

### 📊 Loading & Preprocessing of Raw / Pre‑computed Data
- **`load_prepare.m`**
  - Loads `X_train`, `X_test`, labels and subject IDs from the official UCI HAR files
  - Performs basic consistency checks on labels and dimensions

- **`extract_features.m`**
  - Implements **feature extraction from raw Inertial Signals** (your first workflow step)
  - Uses simple signal‑processing operations (FFT, statistics) to compute:
    - 9 signals (`body_acc_x/y/z`, `body_gyro_x/y/z`, `total_acc_x/y/z`)
    - 8 features per signal: mean, std, median, RMS, energy, zero‑crossing rate, and band‑power features
  - Toolbox‑free implementation

### 🔍 Exploratory Data Analysis & Spectrograms
- **`eda.m`**
  - Generates basic EDA plots:
    - Class distribution (train + test)
    - Sample time‑series for selected signals
    - A reference spectrogram for one sample

- **`spectrogram_utils.m`**
  - Unified utilities for spectrogram‑based analysis:
    - `'features'` mode: band‑power features from spectrogram
    - `'save'` mode: save a spectrogram PNG for a given sample

### 🎛️ Feature Selection (Statistical Methods)
- **`feature_selection.m`**
  - Implements several feature selection strategies:
    - ReliefF (when toolbox is available)
    - F‑score (ANOVA‑style) toolbox‑free method
    - PCA
    - Sequential forward selection

### 🤖 Training & Validation of ML Models
- **`train_models.m`**
  - Trains and evaluates **k‑NN** (manual implementation) and **Nearest Centroid** classifiers
  - Uses **K‑fold cross‑validation** with stratified folds
  - Computes accuracy, precision, recall, F1‑score and confusion matrices
  - Does **not require** `tune_k_values.m` by default – it uses a fixed `bestK` unless you explicitly enable tuning with `opts.tuneK = true`.

- **`tune_k_values.m`** (optional)
  - Optional hyperparameter tuning helper for k‑NN
  - Tries a range of odd `k` values with cross‑validation and returns the best
  - Safe to keep in the repo; the pipeline runs fine without calling it (default `opts.tuneK = false`).

### 📈 Evaluation & Visualization
- **`evaluate_models.m`**
  - Produces detailed evaluation reports and confusion matrices
  - Optionally uses MATLAB Toolbox (`confusionmat`, `confusionchart`, `perfcurve`) when available

- **`show_results.m`**
  - Convenience helper to display and save results quickly

- **`generate_confusion_matrices.m`**
  - Re‑creates high‑quality confusion matrix figures for all models in a results struct

### 🚀 Pipeline & Automation
- **`pipeline_run.m`**
  - End‑to‑end runner that glues everything together:
    1. **Loading & preprocessing** of raw data / pre‑computed features
    2. **Feature extraction** from raw signals
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

## MATLAB Toolboxes and Functions

This project can benefit from MATLAB toolboxes, but does **not strictly depend** on them:

- **Statistics and Machine Learning Toolbox**
  - `confusionmat`, `confusionchart` – professional confusion matrices
  - `cvpartition`, `kfoldPredict` – K‑fold cross‑validation helpers
  - `relieff`, `pca`, `sequentialfs` – feature selection and dimensionality reduction

- **Signal Processing Toolbox**
  - `spectrogram` – time‑frequency analysis and spectrogram visualization

- **Deep Learning Toolbox** (optional)
  - On some MATLAB setups, `confusionchart` and related plotting utilities are provided here.

When these toolboxes are not available, the project automatically falls back to:

- F‑score based feature selection (`feature_selection.m`)
+- Manual confusion matrix and metric computation (`train_models.m`)
- Toolbox‑free STFT and spectrogram plotting (`regenerate_results.m`, `spectrogram_utils.m`)

### Feature Selection Algorithm
- **`feature_selection.m`** - ReliefF, F-score, PCA, Sequential FS

### ML Strategy (Models and strategy)

- **Models**
  - **k‑Nearest Neighbors (k‑NN)**:
    - A simple, interpretable baseline commonly used for HAR tasks.
    - The project includes both a **manual** (toolbox‑free) implementation and an optional `fitcknn`-based version.
  - **Nearest Centroid Classifier**:
    - Uses the mean feature vector per class as the prototype.
    - Fast and interpretable, especially useful for low‑dimensional feature sets.

- **Evaluation and strategy**
  - **K‑fold cross‑validation** (`train_models.m`, `tune_k_values.m`):
    - The dataset is split into K folds; each fold is used once for testing and K‑1 times for training.
    - Stratified sampling is used to preserve class balance.
  - **Hyperparameter tuning**:
    - `tune_k_values.m` tries a range of odd `k` values using cross‑validation and selects the best performing one.
  - **Metrics**:
    - Accuracy, precision, recall, F1‑score and confusion matrix.

This section can be used in presentations to answer: “How was model selection performed?”, “Why k‑NN?”, and “Why did we use cross‑validation?”.

### GUI (MATLAB App Designer) ve Görselleştirmeler

-- **GUI – `create_app_clean.m`**
  - A MATLAB App Designer application that allows the user to:
    - Select the dataset,
    - Display the selected sample's time series and spectrogram,
    - Run activity prediction using the trained k‑NN model,
    - View prediction results and class probabilities.
  - Suitable for live demonstrations: data selection → signal display → model prediction → result.

- **Visualizations**
- **`eda.m`** - EDA grafikleri
- **`evaluate_models.m`** - Confusion matrix (MATLAB Toolbox ile), ROC curves
-- **`generate_confusion_matrices.m`** - Create professional confusion matrix figures (with activity labels)
- **`create_app_clean.m`** - GUI grafikleri (raw signals, normalized features)
- **`show_results.m`** - Sonuç görselleştirmeleri

**Confusion Matrix Features:**
- Use of MATLAB `confusionchart` (Statistics/Deep Learning Toolbox)
- Professional styling with a sky/blue colormap
- Automatic loading of activity labels (`WALKING`, `WALKING_UPSTAIRS`, etc.)
- Numeric values displayed inside matrix cells
- Save as high‑quality PNG and FIG formats

---

## Example Result Figures (`results_figs/`)

Once you run the pipeline or `regenerate_results.m`, the following figures are saved under **`human+activity+recognition+using+smartphones/results_figs/`**.  
Relative paths are used so that they render directly on GitHub.

- **Class distribution**

  ![Class distribution](human+activity+recognition+using+smartphones/results_figs/class_distribution.png)

- **Sample time series (body\_acc\_x/y/z)**

  ![Sample time series](human+activity+recognition+using+smartphones/results_figs/sample_time_series.png)

- **Spectrogram – sample 3676**

  ![Sample spectrogram](human+activity+recognition+using+smartphones/results_figs/sample_spectrogram.png)

- **k‑NN / Nearest Centroid confusion matrix**

  ![kNN confusion matrix](human+activity+recognition+using+smartphones/results_figs/knn_confusion_matrix.png)

All these figures can be regenerated at any time via `regenerate_results.m` or as part of the full `pipeline_run` workflow.

---

## Quick Start

### 1. Launch the GUI (App Designer)
```matlab
app = create_app_clean();
```

### 2. Feature Extraction (from raw signals)
```matlab
[F, L] = extract_features('UCI HAR Dataset');
```

### 3. Feature Selection
```matlab
optsFS.numFeatures = 50;
[selIdx, selNames, Fsel] = feature_selection(F, L, 'relieff', optsFS);
```

### 4. (Optional) Hyperparameter Tuning for k‑NN
```matlab
[bestK, kStats] = tune_k_values(Fsel, L);
```

### 5. Model Training & Validation
```matlab
opts = struct('bestK', bestK, 'K', 5, 'tuneK', false);
results = train_models(Fsel, L, opts);
```

### 6. End‑to‑End Pipeline (your full workflow)
```matlab
pipeline_run('UCI HAR Dataset', struct('saveResults', true, 'summaryOnly', false));
```

### 7. Regenerate Presentation Figures (EDA + spectrogram + confusion matrix)

```matlab
cd human+activity+recognition+using+smartphones
regenerate_results;   % class_distribution, sample_time_series, sample_spectrogram, knn_confusion_matrix
```

### 8. Generate Confusion Matrices for All Models
```matlab
% Confusion matrix'leri oluştur ve kaydet (activity label'ları ile)
generate_confusion_matrices('UCI HAR Dataset');

% Veya mevcut results'tan oluştur:
generate_confusion_matrices('UCI HAR Dataset', struct('loadFromFile', true));
```

---

## Conclusion

This project demonstrates a complete, **practical HAR workflow** on the UCI dataset, from raw signal loading and preprocessing, through signal‑processing‑based feature extraction and statistical feature selection, to training and validating classic machine learning models (k‑NN and Nearest Centroid).  
The resulting figures (EDA, spectrograms, confusion matrices) and the App Designer GUI make it suitable both as a **teaching resource** and as a **presentation‑ready demo**.  
With toolbox‑free fallbacks and clear separation of steps, the code can be adapted or extended to new sensors, additional activities, or alternative models with minimal changes.

---

## Output Files

- **`human+activity+recognition+using+smartphones/results_figs/`** – all figures (EDA, spectrograms, confusion matrices, etc.)
- **`human+activity+recognition+using+smartphones/har_models_results.mat`** – saved model metrics
- **`human+activity+recognition+using+smartphones/knn_model.mat`** – saved k‑NN model
