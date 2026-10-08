# Bank Marketing — Machine Learning Classification

A machine learning project focused on predicting whether a bank customer will subscribe to a term deposit following a marketing campaign.

The project uses real-world data from the UCI Machine Learning Repository and explores the complete machine learning workflow, from exploratory data analysis to model training, evaluation, and interpretation.

## Project Objectives

- Explore and understand a real-world dataset.
- Identify relevant features and potential data quality issues.
- Build a baseline classification model.
- Apply appropriate data preprocessing techniques.
- Train and compare different machine learning algorithms.
- Evaluate model performance using suitable classification metrics.
- Interpret predictions and identify important features.
- Build a reproducible machine learning pipeline.

**Key consideration:** The goal is to predict customer responses using information available before a marketing call. Features that could introduce data leakage, such as call duration, will be excluded from the main prediction pipeline.

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The dataset contains information about direct marketing campaigns conducted by a Portuguese banking institution.

- **Task:** Binary Classification
- **Target:** Whether a customer subscribes to a term deposit (`yes` / `no`)
- **Features:** Customer demographics, financial information, and previous campaign interactions

The dataset is provided by the UCI Machine Learning Repository under the CC BY 4.0 license.

## Tech Stack

- **Language:** Python 3.12
- **Data Analysis:** pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** scikit-learn
- **Experimentation:** Jupyter Notebook
- **Development Environment:** PyCharm, Ubuntu (WSL2)
- **Version Control:** Git & GitHub

## Project Structure

```text
bank-marketing-ml/
├── data/
│   └── raw/             # Original datasets
├── notebooks/
│   └── 01_eda.ipynb     # Exploratory data analysis
├── src/                 # ML pipeline and prediction code
├── models/              # Trained models
├── reports/
│   └── figures/         # Visualizations and results
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.12
- Git
- uv (recommended for dependency management)

### Installation

Clone the repository and navigate to the project directory.

Create a virtual environment:

```bash
uv venv --python 3.12 .venv
```

Install the required dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Open the project in PyCharm or launch Jupyter using the configured Python environment.

## Machine Learning Workflow

The planned development workflow includes:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing and feature selection
3. Train/test split and baseline model
4. Model training and comparison
5. Performance evaluation
6. Model interpretation
7. Model serialization and inference

Evaluation will consider metrics such as Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

## Results

*To be added after model training and evaluation.*

This section will include model comparisons, evaluation metrics, relevant visualizations, and a discussion of limitations.

## Project Status

**In progress — Environment setup completed.**

The machine learning implementation and experimental results will be added incrementally.

## Acknowledgements

Dataset: [Bank Marketing — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.*

## License

The project source code is released under the MIT License. The original dataset is subject to its own CC BY 4.0 license and attribution requirements.