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
    # **UCI HAR – Human Activity Recognition Using Smartphones**

    ## **Overview**

    This repository presents an end-to-end **Human Activity Recognition (HAR)** pipeline implemented entirely in **MATLAB**, using the well-known **UCI HAR** dataset.
    The objective is to automatically classify six human activities using inertial sensor signals (3-axis accelerometer and gyroscope) collected from a smartphone worn at the waist:

    * **WALKING**
    * **WALKING_UPSTAIRS**
    * **WALKING_DOWNSTAIRS**
    * **SITTING**
    * **STANDING**
    * **LAYING**

    The project implements a complete workflow, including:

    * **Loading and preprocessing of raw data**
    * **Signal-processing-based feature extraction**
    * **Statistical feature selection**
    * **Training and validation of machine learning models**
    * **Visualization of results and a MATLAB GUI for demonstration**

    An interactive **MATLAB App Designer GUI** (`create_app_clean.m`) is provided for real-time visualization, signal inspection, and classification using the trained k-NN model.

    MATLAB toolboxes enhance the workflow when available—particularly **Statistics and Machine Learning Toolbox** and **Signal Processing Toolbox**—while **toolbox-free fallback implementations** ensure the pipeline remains functional in minimal MATLAB environments.

    ---

    ## **Dataset Description**

    * **Source:** UCI Machine Learning Repository — *Human Activity Recognition Using Smartphones Dataset*
    * **Device:** Samsung Galaxy S smartphone worn on the waist
    * **Sensors:**

      * 3-axis accelerometer (total + body acceleration)
      * 3-axis gyroscope
    * **Sampling Rate:** 50 Hz
    * **Windowing:**

      * 2.56-second windows (128 samples)
      * Each window includes 9 time-series signals
    * **Pre-computed features:**

      * `X_train.txt`, `X_test.txt` contain ≈560 features
      * `y_train.txt`, `y_test.txt` contain 6 activity labels

    This project utilizes:

    1. **Pre-computed feature matrices** for rapid training and baselines
    2. **Raw “Inertial Signals”** for custom feature extraction and spectrogram-based analysis

    ---

    ## **Project Structure**

    ```
    UCI HAR Dataset/
    results_figs/
        class_distribution.png
        sample_time_series.png
        sample_spectrogram.png
        knn_confusion_matrix.png
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

    This structure enables:

    * Full pipeline execution via **`pipeline_run`**
    * Reproduction of all presentation figures via **`regenerate_results`**
    * Launching a graphical demo via **`create_app_clean`**

    ---

    ## **Key Components**

    ### **1. GUI Demo – `create_app_clean.m`**

    A MATLAB App Designer application supporting:

    * Raw and processed signal visualization
    * Interactive selection of samples
    * Real-time activity classification with the saved k-NN model
    * Visual display of probabilities and predicted labels

    Designed for teaching, demonstrations, and live presentations.

    ---

    ### **2. Data Loading & Preprocessing**

    * **`load_prepare.m`**
      Loads all UCI HAR feature sets, labels, and subject identifiers.

    * **`extract_features.m`**
      Implements **toolbox-free feature extraction** from raw Inertial Signals:

      * Mean, standard deviation, median
      * RMS
      * Signal energy
      * Zero-crossing rate
      * Band-power features
        Produces a feature matrix aligned with corresponding activity labels.

    ---

    ### **3. Exploratory Data Analysis & Spectrograms**

    * **`eda.m`**
      Generates dataset distribution plots, time-series examples, and a reference spectrogram.

    * **`spectrogram_utils.m`**
      Provides unified utilities for spectrogram generation and feature computation.

    ---

    ### **4. Feature Selection**

    * **`feature_selection.m`**
      Supports multiple selection strategies:

      * ReliefF (toolbox)
      * F-score (toolbox-free)
      * PCA
      * Sequential forward selection

    ---

    ### **5. Machine Learning Models**

    * **k-Nearest Neighbors (k-NN)**
      Implemented both manually and via MATLAB’s `fitcknn` when available.
      Serves as an interpretable, dataset-appropriate classifier.

    * **Nearest Centroid Classifier**
      A lightweight baseline model using class-mean prototypes.

    Both models are evaluated using:

    * **Stratified K-fold cross-validation**
    * **Accuracy, precision, recall, F1-score, and confusion matrices**

    ---

    ### **6. Model Evaluation & Visualization**

    * **`evaluate_models.m`**
      Computes evaluation metrics and generates confusion matrices.

    * **`generate_confusion_matrices.m`**
      Produces publication-quality confusion matrices with activity labels.

    * **`show_results.m`**
      Provides convenient visualization and saving of result summaries.

    ---

    ### **7. Automation & Reproducibility**

    * **`pipeline_run.m`**
      Executes the entire workflow from data loading to final evaluation.

    * **`regenerate_results.m`**
      Reproduces all figures stored in `results_figs/`:

      * Class distribution
      * Sample time series
      * Reference spectrogram
      * k-NN confusion matrix

    ---

    ## **Figures**

    All figures generated by the project are stored in:

    `human+activity+recognition+using+smartphones/results_figs/`

    ![Class distribution](results_figs/class_distribution.png)

    ![Sample time series](results_figs/sample_time_series.png)

    ![Sample spectrogram](results_figs/sample_spectrogram.png)

    ![kNN confusion matrix](results_figs/knn_confusion_matrix.png)

    These figures can be reproduced at any time using the scripts mentioned above.

    ---

    ## **Quick Start**

    ### **Launch the GUI**

    ```matlab
    app = create_app_clean();
    ```

    ### **Extract Features**

    ```matlab
    [F, L] = extract_features('UCI HAR Dataset');
    ```

    ### **Feature Selection**

    ```matlab
    optsFS.numFeatures = 50;
    [selIdx, selNames, Fsel] = feature_selection(F, L, 'relieff', optsFS);
    ```

    ### **Optional: Hyperparameter Tuning**

    ```matlab
    [bestK, kStats] = tune_k_values(Fsel, L);
    ```

    ### **Model Training**

    ```matlab
    opts = struct('bestK', bestK, 'K', 5, 'tuneK', false);
    results = train_models(Fsel, L, opts);
    ```

    ### **Full Pipeline**

    ```matlab
    pipeline_run('UCI HAR Dataset', struct('saveResults', true, 'summaryOnly', false));
    ```

    ### **Recreate Presentation Figures**

    ```matlab
    cd human+activity+recognition+using+smartphones
    regenerate_results;
    ```

    ### **Generate Confusion Matrices**

    ```matlab
    generate_confusion_matrices('UCI HAR Dataset');
    ```

    ---

    ## **Conclusion**

    This repository delivers a complete and reproducible **HAR analysis framework** in MATLAB.
    The workflow integrates raw-signal processing, feature engineering, traditional ML classification, cross-validated evaluation, and fully automated visualization.
    The inclusion of a MATLAB GUI makes the project suitable for:

    * Educational use
    * Research comparisons
    * Live demonstrations
    * Rapid prototyping

    Toolbox-free alternatives allow the pipeline to operate on a wide range of MATLAB installations while maintaining high interpretability and methodological clarity.

    ---

    ## **Output Files**

    * **results_figs/** — visualizations (EDA, spectrograms, confusion matrices)
    * **har_models_results.mat** — stored model metrics
    * **knn_model.mat** — trained k-NN classifier

    ---

    If a shorter or more academic version is needed, it can be provided as well.
