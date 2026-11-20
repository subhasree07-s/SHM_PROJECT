# StructureGuard
### Deep Learning-Based Multi-Modal Structural Health Monitoring

StructureGuard is an experimental structural health monitoring framework that combines vibration sensor analysis, visual crack detection, and thermal hotspot detection. It uses three independently trained classification models and combines their abnormality scores into a **Structural Stress Index (SSI)** for scenario-based assessment.

> **Research limitation:** The three modalities were trained using independent datasets, not synchronised measurements from the same physical structures. The fusion workflow is therefore a proof of concept. SSI is an experimental composite abnormality index—not a physical stress measurement, calibrated failure probability, or engineering safety certification.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Models and Datasets](#models-and-datasets)
- [Structural Stress Index](#structural-stress-index)
- [Experimental Results](#experimental-results)
- [Dashboard](#dashboard)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Execution](#installation-and-execution)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Before Publishing](#before-publishing)

## Overview

Structural damage can manifest in different ways. Vibration measurements can indicate changes in structural behaviour, visual images can reveal surface cracks, and thermal images can show abnormal temperature patterns.

StructureGuard explores these complementary data sources through three independent classification pipelines:

1. **Vibration analysis:** classifies sensor readings as Normal, Minor Damage, or Severe Damage.
2. **Visual analysis:** classifies images as Crack or No Crack.
3. **Thermal analysis:** classifies thermal images as Hotspot or Normal.

The model outputs are converted into abnormality scores and combined using an experimental SSI. A Streamlit dashboard provides individual predictions, model evaluation results, scenario-based fusion, and report downloads.

## Key Features

- Vibration-based condition classification using Random Forest.
- Image-based crack detection using a Convolutional Neural Network (CNN).
- Thermal hotspot detection using a CNN.
- Score-level fusion using an equal-weight SSI.
- Safe, Moderate Risk, and High Risk categories.
- Interactive Streamlit dashboard.
- Model evaluation views, including classification metrics and confusion matrices.
- Individual modality inference and scenario-based multimodal assessment.
- Downloadable report outputs.

## System Architecture

```text
                   Input Data
                       |
       +---------------+---------------+
       |               |               |
   Vibration         Visual          Thermal
   Readings          Image            Image
       |               |               |
  Random Forest        CNN             CNN
       |               |               |
  Normal / Minor   Crack / No       Hotspot /
  / Severe Damage    Crack           Normal
       |               |               |
       +---------------+---------------+
                       |
             Abnormality Scores
                       |
              Equal-Weight Fusion
                       |
          Structural Stress Index (SSI)
                       |
             Safe / Moderate / High
                       |
              Streamlit Dashboard
```

The models are trained independently. Fusion combines their abnormality scores for a user-provided scenario; it does not establish that the three inputs came from the same physical structure.

## Models and Datasets

### 1. Vibration Model

- **Algorithm:** Random Forest Classifier
- **Input features:** `Accel_X`, `Accel_Y`, `Accel_Z`, `Strain`, and `Temperature`
- **Output classes:** Normal, Minor Damage, Severe Damage
- **Final configuration:** 300 trees with balanced class weights
- **Evaluation:** Expanding-window chronological cross-validation with a purge gap, followed by evaluation on the designated test block.

The vibration dataset contains 1,000 records:

| Condition | Samples |
|---|---:|
| Normal | 704 |
| Minor Damage | 183 |
| Severe Damage | 113 |
| **Total** | **1,000** |

### 2. Visual Crack Detection Model

- **Algorithm:** Convolutional Neural Network (CNN)
- **Input:** RGB images resized to 128 × 128 pixels
- **Output classes:** Crack, No Crack
- **Architecture:** Convolutional layers with 32, 64, and 128 filters, followed by pooling, a dense layer, dropout, and a sigmoid output.
- **Training:** Adam optimiser, binary cross-entropy loss, and early stopping.

The visual dataset contains 4,000 images: 2,000 images in each class.

### 3. Thermal Hotspot Detection Model

- **Algorithm:** Convolutional Neural Network (CNN)
- **Input:** RGB images resized to 128 × 128 pixels
- **Output classes:** Hotspot, Normal
- **Architecture:** Convolutional layers with 32, 64, and 128 filters, followed by global average pooling, a dense layer, dropout, and a sigmoid output.
- **Training:** Adam optimiser, binary cross-entropy loss, class weighting, and early stopping.

The thermal dataset contains 433 images:

| Split | Hotspot | Normal |
|---|---:|---:|
| Training | 272 | 74 |
| Validation | 68 | 19 |
| **Total** | **340** | **93** |

## Structural Stress Index

StructureGuard converts the three model outputs into abnormality scores:

- **Vibration:** probability of Minor Damage plus probability of Severe Damage.
- **Visual:** probability of Crack.
- **Thermal:** probability of Hotspot.

Let `P_V` represent vibration abnormality, `P_I` represent visual crack probability, and `P_T` represent thermal hotspot probability.

The SSI is calculated using equal weights:

```text
SSI = 100 × (w_V × P_V + w_I × P_I + w_T × P_T)

w_V = w_I = w_T = 1/3

SSI = (100/3) × (P_V + P_I + P_T)
```

### Experimental Risk Categories

| SSI Range | Category |
|---|---|
| 0–30 | Safe |
| Greater than 30 and up to 70 | Moderate Risk |
| Greater than 70 and up to 100 | High Risk |

The weights and thresholds are experimental choices. They have not been calibrated against physical structural measurements or validated engineering standards. SSI must not be interpreted as a direct measurement of stress or a probability of structural failure.

## Experimental Results

The following results were obtained during the reported experiments. The evaluation split is stated for each model because the tasks and datasets differ.

| Model | Evaluation split | Accuracy | Macro F1-score |
|---|---|---:|---:|
| Vibration Random Forest | Test | 69.93% | 0.6024 |
| Visual CNN | Validation | 99.00% | 0.9900 |
| Thermal CNN | Validation | 89.66% | 0.8564 |

**Interpretation:**
- The vibration model had difficulty identifying Minor Damage; its test F1-score for that class was 0.29.
- Visual and thermal results are validation metrics, not test-set metrics.
- The visual model's high validation score should be interpreted cautiously until duplicate and near-duplicate image checks and independent testing are completed.
- These metrics should not be compared as though all models solve the same task or use the same evaluation split.

## Dashboard

The Streamlit application contains five modules:

1. **Project Overview:** presents the system, datasets, architecture, and key metrics.
2. **Model Evaluation:** displays evaluation metrics, confusion matrices, class-wise performance, and training curves.
3. **Individual Modality Analysis:** accepts vibration readings and uploaded visual or thermal images for individual predictions.
4. **Multimodal Fusion:** combines abnormality scores and displays the resulting SSI and risk category.
5. **Reports and Insights:** presents insights from the latest session result and provides report downloads.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Machine learning | Scikit-learn |
| Deep learning | TensorFlow / Keras |
| Data processing | Pandas, NumPy |
| Image processing | Pillow |
| Dashboard | Streamlit |
| Visualisation | Plotly, Streamlit charts |
| Model persistence | Joblib, Keras model files |

## Project Structure

This is a guide to the main paths used by the project. Update it to match the files included in your GitHub repository.

```text
StructureGuard/
├── data/
│   ├── VIBRATIONAL/
│   ├── VISUAL/
│   └── THERMAL/
├── results/
│   ├── vibration/
│   │   └── final/
│   ├── visual/
│   └── thermal/
├── src/
│   ├── models/
│   │   └── run_multimodal_fusion.py
│   └── ui/
│       └── app.py
├── README.md
└── requirements.txt
```

Your local repository may contain additional training scripts, preprocessing utilities, configuration files, and evaluation outputs. Add them to this tree after confirming their actual names and locations.

## Installation and Execution

### Prerequisites

- Python environment compatible with the TensorFlow version used to train the models.
- Git.
- Trained model files and any required input files.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd StructureGuard
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository includes `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you have not created it yet, the main packages used by the application include:

```bash
pip install streamlit tensorflow scikit-learn pandas numpy pillow plotly joblib
```

For reproducibility, record and pin the package versions from the environment in which the models were trained and the dashboard was tested.

### 4. Run the dashboard

From the repository root:

```bash
streamlit run src/ui/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser.

**Note:** The application expects trained models and required input files at the paths referenced in the source code. Update those paths if your repository layout differs.

## Limitations

- The three modalities use independently collected datasets; synchronised measurements from the same physical structures were not available.
- Fusion is a scenario-based proof of concept and has not been validated using real-world synchronised multimodal observations.
- SSI weights and risk thresholds are experimental and have not been calibrated against physical structural measurements or engineering standards.
- The vibration model has weaker performance on the Minor Damage class.
- The visual CNN's high validation performance requires further verification through duplicate-image auditing and independent testing.
- Model probabilities have not been calibrated against independently measured physical damage.
- Dashboard report data is session-based and may not persist after a session reset or application restart.

## Future Work

- Build a synchronised multimodal dataset collected from the same structures.
- Improve the vibration model's ability to distinguish minor damage.
- Evaluate visual and thermal models on independent test datasets and conduct duplicate-image checks.
- Investigate temporal deep learning and alternative feature representations for vibration data.
- Explore learned or adaptive multimodal fusion methods.
- Calibrate SSI weights and risk thresholds using physical measurements and expert assessment.
- Validate the complete framework in practical structural monitoring environments.

## Before Publishing

1. Add dataset source links and licence information for all three datasets.
2. Check whether each dataset licence permits redistribution. If not, do not upload raw datasets; provide source links and acquisition instructions instead.
3. Include trained model files only if their size and applicable licences permit redistribution. For large files, consider Git LFS or a model-hosting service.
4. Include a `requirements.txt` with tested package versions.
5. Remove local absolute paths, personal information, temporary files, and machine-specific configuration.
6. Include sample inputs only if you have permission to share them.
7. Verify that a fresh clone can install dependencies and launch the dashboard.

---

## Project

**StructureGuard — Deep Learning-Based Multi-Modal Structural Health Monitoring**

An academic project exploring the integration of vibration, visual, and thermal analysis for structural health monitoring.
