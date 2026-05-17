# Apple Price Forecasting Benchmarking Framework

## Overview

This repository provides the reproducibility framework accompanying the research study:

**“Multi-Model Benchmarking for Apple Price Forecasting Across Production and Terminal Markets”**

The framework evaluates statistical, machine learning, deep learning, and transformer-based forecasting models for apple price prediction across multiple production and terminal markets under a leakage-aware recursive forecasting setup.

The repository is intended to support:
- experimental reproducibility,
- benchmarking transparency,
- methodological validation,
- forecasting workflow understanding,
- deployment reproducibility.

---

## Important Repository Note

This repository is released specifically for:

- reproducibility support,
- academic verification,
- methodological transparency,
- deployment workflow demonstration.

It does **not** contain the complete proprietary production framework, confidential deployment infrastructure, or internal operational optimization components used in the full institutional forecasting system.

Certain:
- deployment utilities,
- confidential integrations,
- internal automation modules,
- operational dashboards,
- institution-specific configurations,
- proprietary optimization routines

have intentionally been excluded.

The repository therefore represents a:
> **research reproducibility implementation**
rather than the complete production-grade institutional forecasting platform.

---

## Research Objectives

The framework was developed to:

- benchmark forecasting models across agricultural markets,
- evaluate recursive multi-step forecasting behavior,
- compare forecasting stability under seasonal dynamics,
- assess model generalization across heterogeneous markets,
- support operational agricultural decision systems.

---

## Forecasting Models

### Statistical Models
- TBATS

### Machine Learning Models
- Random Forest
- LightGBM

### Deep Learning Models
- LSTM
- Transformer
- N-BEATS
- N-HiTS
- PatchTST
- NeuralProphet

### Ensemble Models
- Weighted Ensemble Framework

---

## Key Methodological Features

- Leakage-aware seasonal validation
- Recursive multi-step forecasting
- Mask-aware forecasting pipeline
- Rolling seasonal evaluation
- Statistical significance testing
- Cross-market benchmarking
- Recursive deployment inference
- Conformal prediction compatibility
- Experiment tracking and logging

---

## Repository Structure

```text
APPLE-PRICE-FORECASTING-DEPLOYMENT/
│
├── data/
├── deployment/
├── experiments/
│
├── preprocess.py
├── train.py
├── evaluate.py
│
├── requirements.txt
├── environment.yml
├── README.md
├── LICENSE
└── .gitignore
Dataset

The repository contains processed forecasting-ready datasets derived from apple mandi market records.

The processed datasets include:

daily timestamps,
average prices,
seasonal masks,
forecasting-ready continuous calendar structures.

Raw institutional operational datasets are not included.

Installation
Clone Repository
git clone https://github.com/your-repository/apple-price-forecasting.git

cd APPLE-PRICE-FORECASTING-DEPLOYMENT
Create Environment

Using Conda:

conda env create -f environment.yml

conda activate apple-price-forecasting

Or using pip:

pip install -r requirements.txt
Preprocessing

Run preprocessing pipeline:

python preprocess.py

The preprocessing module:

validates processed datasets,
performs scaling,
generates forecasting sequences,
prepares leakage-aware train-validation splits.
Training

Run training pipeline:

python train.py

The training framework implements:

recursive forecasting,
leakage-aware validation,
seasonal evaluation,
multi-model benchmarking,
experiment logging.
Evaluation

Run evaluation pipeline:

python evaluate.py

The evaluation framework computes:

RMSE,
MAE,
RMSLE,
MASE,
sMAPE,
R²,
Directional Accuracy,
Theil’s U statistics,
conformal intervals,
comparative statistical metrics.
Experimental Outputs

Generated outputs include:

experiments/

Containing:

trained model artifacts,
forecasting outputs,
prediction tables,
metrics,
logs,
visualization artifacts.
Reproducibility Statement

The repository includes:

reproducibility-oriented implementations,
forecasting pipelines,
processed datasets,
evaluation workflows,
experiment management structure.

The implementation preserves the core experimental methodology used in the manuscript to support reproducibility of reported benchmarking results.

Confidentiality Notice

Some components from the full institutional forecasting ecosystem have intentionally been excluded, including:

production APIs,
deployment orchestration,
internal dashboards,
operational monitoring tools,
institutional automation systems,
proprietary optimization routines.

The repository should therefore be interpreted as:

an academic reproducibility framework rather than a full operational deployment release.

Intended Use

This repository is intended for:

academic research,
forecasting benchmarking,
reproducibility validation,
educational use,
methodological comparison studies.
Citation

If you use this repository or methodology, please cite the associated manuscript.

Author(s). Multi-Model Benchmarking for Apple Price Forecasting Across Production and Terminal Markets.
License

This repository is released under the MIT License.

Contact

For academic collaboration, reproducibility questions, or methodological discussions, please contact the repository authors.