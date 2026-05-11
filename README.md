# Machine Learning Regression and Image Classification Project

## Project Objectives

This project implements a complete university machine learning workflow using only the two datasets placed in this project:

- A numerical regression dataset in `datasets/numerical/`
- An image classification dataset in `datasets/images/`

The project trains and evaluates four models:

- Linear Regression for numerical regression
- KNN Regressor for numerical regression
- Logistic Regression for image classification
- KMeans for image clustering-based classification using majority-vote cluster-to-label mapping

## Datasets Used

The code does not download, generate, or substitute datasets. It uses only the files already placed in the project dataset folders.

Numerical dataset:

- Expected location: `datasets/numerical/`
- Current configured file: `salaries_cyber.csv`
- Configured target column: `salary_in_usd`
- Configured leakage/drop columns: `salary`, `salary_currency`

Image dataset:

- Expected location: `datasets/images/`
- Expected layout: `datasets/images/class_name/image_file.jpg`
- Maximum classes used: 5
- Current detected classes: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`
- Configured image size: `128x128`

All important dataset settings are located in `config.py`.

## Folder Structure

```text
ML_Project/
├── config.py
├── datasets/
│   ├── numerical/
│   └── images/
├── notebooks/
│   ├── numerical_regression.ipynb
│   └── image_classification.ipynb
├── src/
│   ├── numerical/
│   ├── image/
│   └── common/
├── outputs/
│   ├── figures/
│   ├── confusion_matrices/
│   ├── reports/
│   └── trained_models/
├── docs/
│   ├── Project_Cover_Sheet.docx
│   ├── Project_Description.docx
│   ├── Project_Description.pdf
│   └── README.md
├── requirements.txt
└── main.py
```

## Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How To Run

Run the full project pipeline:

```powershell
python main.py
```

This command trains all four models, evaluates them, saves plots and reports, saves trained models, and rebuilds the Word/PDF project documents.

Run the graphical user interface:

```powershell
python gui_app.py
```

The GUI loads the trained models from `outputs/trained_models/`. If those files are missing, run `python main.py` first.

## Outputs Produced

The project automatically saves:

- Regression metric summaries in `outputs/reports/`
- Classification metric summaries in `outputs/reports/`
- Predicted-vs-actual plots in `outputs/figures/`
- Residual plots in `outputs/figures/`
- Regression comparison chart in `outputs/figures/`
- Confusion matrix images in `outputs/confusion_matrices/`
- Trained model files in `outputs/trained_models/`
- Final project documents in `docs/`
- Interactive Tkinter GUI through `gui_app.py`

## Results Summary

The exact results are generated from the local datasets when `python main.py` is executed. The report clearly separates the evaluation logic:

- Regression models use MAE, MSE, RMSE, and R2.
- Classification models use accuracy, confusion matrix, and classification report.

After running the project, open:

- `outputs/reports/complete_results.json`
- `outputs/reports/numerical_regression_report.txt`
- `outputs/reports/image_classification_report.txt`
- `docs/Project_Description.docx`
- `docs/Project_Description.pdf`


