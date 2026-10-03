# End to End Machine Learning Project: Student Performance Prediction

This project predicts a student's **math score** (0–100) from their background and their reading and writing scores. It covers the full machine-learning lifecycle:

- exploring the data in notebooks
- a modular training pipeline that ingests, transforms, and trains
- automatic selection of the best of 7 regression models
- a Flask web app that serves predictions from a form

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline Steps in Detail](#pipeline-steps-in-detail)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Using the Web App](#using-the-web-app)
- [Logging and Error Handling](#logging-and-error-handling)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Known Limitations and Next Steps](#known-limitations-and-next-steps)

---

## Problem Statement

The project asks how a student's math score relates to these factors:

- gender
- race/ethnicity
- parents' level of education
- lunch type
- whether they completed a test preparation course
- reading score
- writing score

The goal is a model that predicts the math score from these inputs, served through a simple web interface.

## Dataset

- **Source:** [Students Performance in Exams (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Location in repo:** `notebook/data/stud.csv`
- **Size:** 1,000 rows × 8 columns, no missing values

| Column | Type | Values |
|---|---|---|
| `gender` | categorical | `female`, `male` |
| `race_ethnicity` | categorical | `group A` – `group E` |
| `parental_level_of_education` | categorical | `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree` |
| `lunch` | categorical | `standard`, `free/reduced` |
| `test_preparation_course` | categorical | `none`, `completed` |
| `reading_score` | numeric | 0–100 |
| `writing_score` | numeric | 0–100 |
| **`math_score`** | numeric (**target**) | 0–100 |

---

## Architecture

```
                         ┌──────────────────────── TRAINING ────────────────────────┐
                         │                                                           │
 notebook/data/stud.csv ─┼─► 1. Data Ingestion ──► 2. Data Transformation ──► 3. Model Trainer
                         │      (split 80/20)         (impute, scale, encode)     (tune 7 models,
                         │          │                        │                     pick best)
                         │          ▼                        ▼                        │
                         │   artifacts/train.csv     artifacts/preprocessor.pkl       ▼
                         │   artifacts/test.csv                              artifacts/model.pkl
                         │   artifacts/data.csv                                       │
                         └───────────────────────────────────────────────────────────┼─┘
                                                                                     │
                         ┌──────────────────────── PREDICTION ──────────────────────┼─┐
                         │                                                           │ │
   Browser form ────────►│ Flask app.py ──► CustomData ──► PredictPipeline ◄─────────┘ │
   (/predictdata)        │                 (form → DataFrame)   │ loads preprocessor   │
         ▲               │                                      │ + model, predicts    │
         └───────────────┼──────────── predicted math score ◄───┘                      │
                         └─────────────────────────────────────────────────────────────┘
```

The project has three layers:

1. **Components** (`src/components/`): one class for each stage of the ML workflow. Each class reads its settings from a small `@dataclass` config.
2. **Pipelines** (`src/pipeline/`):
   - `train_pipeline.py` runs the components in order.
   - `predict_pipeline.py` loads the saved artifacts for inference.
3. **Serving** (`app.py` + `templates/`): a Flask web app that collects input through an HTML form and returns a prediction.

Shared helpers sit in `src/`:

- `utils.py`: saving and loading objects, and evaluating models
- `logger.py`: logging setup
- `exception.py`: the custom exception class

---

## Project Structure

```
mlproject/
├── app.py                          # Flask web app (entry point for serving)
├── setup.py                        # Makes `src` installable as the `mlproject` package
├── requirements.txt                # Python dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── exception.py                # CustomException with file name + line number
│   ├── logger.py                   # Writes timestamped log files to logs/
│   ├── utils.py                    # save_object, load_object, evaluate_models
│   │
│   ├── components/
│   │   ├── data_ingestion.py       # Step 1: read CSV, train/test split
│   │   ├── data_transformation.py  # Step 2: preprocessing pipeline
│   │   ├── model_trainer.py        # Step 3: tune models, save the best one
│   │   ├── model_evaluations.py    # (placeholder, empty)
│   │   └── model_validation.py     # (placeholder, empty)
│   │
│   └── pipeline/
│       ├── train_pipeline.py       # Runs steps 1 → 2 → 3
│       └── predict_pipeline.py     # CustomData + PredictPipeline for inference
│
├── templates/
│   ├── index.html                  # Landing page ( / )
│   └── home.html                   # Prediction form ( /predictdata )
│
├── artifacts/                      # Generated by training
│   ├── data.csv                    # Copy of the raw dataset
│   ├── train.csv / test.csv        # 80/20 split
│   ├── preprocessor.pkl            # Fitted ColumnTransformer
│   └── model.pkl                   # Best trained model
│
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb   # Exploratory data analysis
│   ├── 2. MODEL TRAINING.ipynb              # Model experiments
│   └── data/stud.csv                        # Raw dataset
│
└── logs/                           # Runtime log files (git-ignored)
```

---

## Pipeline Steps in Detail

### 1. Data Ingestion (`src/components/data_ingestion.py`)

1. Reads `notebook/data/stud.csv` into a pandas DataFrame.
2. Saves a raw copy to `artifacts/data.csv`.
3. Splits the data 80/20 into train and test sets (`random_state=42`, so the split is the same every run).
4. Writes `artifacts/train.csv` and `artifacts/test.csv`, and returns their paths.

### 2. Data Transformation (`src/components/data_transformation.py`)

This step builds a scikit-learn `ColumnTransformer` with two sub-pipelines:

| Columns | Steps |
|---|---|
| Numerical: `reading_score`, `writing_score` | `SimpleImputer(median)` → `StandardScaler` |
| Categorical: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course` | `SimpleImputer(most_frequent)` → `OneHotEncoder` → `StandardScaler(with_mean=False)` |

Then it:

1. Separates the target column `math_score` from the input features.
2. Fits the preprocessor on the **training data only**, then applies it to both train and test. Fitting on train only prevents data leakage.
3. Saves the fitted preprocessor to `artifacts/preprocessor.pkl`. Prediction reuses this exact transformation.
4. Returns NumPy arrays with the target appended as the last column.

### 3. Model Training (`src/components/model_trainer.py`)

This step trains and tunes 7 regression models with `GridSearchCV` (3-fold cross-validation):

| Model | Hyperparameters searched |
|---|---|
| Linear Regression | none |
| Decision Tree | `criterion` |
| Random Forest | `n_estimators` |
| Gradient Boosting | `learning_rate`, `subsample`, `n_estimators` |
| XGBoost | `learning_rate`, `n_estimators` |
| CatBoost | `depth`, `learning_rate`, `iterations` |
| AdaBoost | `learning_rate`, `n_estimators` |

For each model, `evaluate_models` in `src/utils.py`:

1. finds the best hyperparameters
2. refits the model on the training set
3. scores it with **R²** on the test set

The model with the highest R² is saved to `artifacts/model.pkl`. If no model reaches R² ≥ 0.6, training fails with "No best model found".

### 4. Prediction (`src/pipeline/predict_pipeline.py`)

- **`CustomData`** turns the raw form inputs into a one-row DataFrame. Its column names match the training data.
- **`PredictPipeline.predict()`** loads `preprocessor.pkl` and `model.pkl`, transforms the input, and returns the predicted math score.

### 5. Web App (`app.py`)

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Landing page (`index.html`) |
| `/predictdata` | GET | Shows the prediction form (`home.html`) |
| `/predictdata` | POST | Runs the prediction and shows the result on the same page |

---

## Setup

### Prerequisites

- Python 3.9 or newer (tested on 3.9.6)
- `git`

### 1. Clone the repository

```bash
git clone https://github.com/augustsiu/mlproject.git
cd mlproject
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # installs `src` as a package so imports work from anywhere
```

Dependencies: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `catboost`, `xgboost`, `Flask`, `dill`.

> **macOS note:** XGBoost needs the OpenMP runtime. If importing `xgboost` fails with a `libomp` error, run `brew install libomp`.

---

## Running the Project

Run every command from the project root, with the virtual environment activated.

### Step 1: Train the model

```bash
python -m src.pipeline.train_pipeline
```

This runs ingestion → transformation → training. It takes about 30–60 seconds, because the grid search fits many models. When it finishes it prints the R² score of the best model, for example:

```
Training complete.
R2 score: 0.8804332983749565
```

It also writes `artifacts/preprocessor.pkl` and `artifacts/model.pkl`.

> Run it with `python -m ...` as shown. Running `python src/pipeline/train_pipeline.py` directly fails with `ModuleNotFoundError: No module named 'src'` unless you ran `pip install -e .`.

> You may see many `RuntimeWarning: ... encountered in matmul` messages during training. They come from a known numpy issue on macOS and are harmless.

The repo already includes trained `model.pkl` and `preprocessor.pkl` files, so you can skip this step and go straight to the web app. Retrain whenever you change the data, the preprocessing, or the models.

### Step 2: Start the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5001/predictdata** in your browser.

The app runs on port **5001**, because macOS's AirPlay Receiver often occupies port 5000. Stop the server with `Ctrl+C`.

### (Optional) Explore the notebooks

```bash
pip install jupyter
jupyter notebook notebook/
```

- `1 . EDA STUDENT PERFORMANCE .ipynb`: data exploration and visualizations
- `2. MODEL TRAINING.ipynb`: model experiments that the production pipeline is based on

---

## Using the Web App

1. Go to http://127.0.0.1:5001/predictdata.
2. Fill in all fields: gender, race/ethnicity, parental education, lunch type, test preparation course, and reading and writing scores (0–100).
3. Click **Predict your Maths Score**.
4. The predicted math score appears below the form.

You can also call the endpoint directly:

```bash
curl -X POST http://127.0.0.1:5001/predictdata \
  --data-urlencode gender=male \
  --data-urlencode "ethnicity=group C" \
  --data-urlencode "parental_level_of_education=some college" \
  --data-urlencode lunch=standard \
  --data-urlencode test_preparation_course=completed \
  --data-urlencode reading_score=80 \
  --data-urlencode writing_score=78
```

> The form field for race/ethnicity is named `ethnicity`. `app.py` maps it to the `race_ethnicity` column.

---

## Logging and Error Handling

- **Logging** (`src/logger.py`): each run creates a timestamped log file in `logs/`, for example `logs/10-03-2026_13-41-00.log`. Pipeline steps log their progress there. Check these files when something goes wrong.
- **Exceptions** (`src/exception.py`): errors in the pipeline are wrapped in `CustomException`. Its message includes the script name and line number where the error happened, for example:

  ```
  Error occurred in python script name [src/pipeline/predict_pipeline.py] line number [13] error message [...]
  ```

---

## Results

On the current dataset and split:

- **Best model:** Linear Regression
- **Test R²:** 0.88

Math scores are strongly linear in the reading and writing scores, so the simple linear model matches or beats the tree-based ensembles.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root with `python -m src.pipeline.train_pipeline`, or run `pip install -e .` |
| `FileNotFoundError` for `artifacts/model.pkl` or `preprocessor.pkl` | Run the training pipeline first (Step 1) |
| `Address already in use` when starting the app | Another process is using port 5001. Stop it, or change the port at the bottom of `app.py` |
| `.venv` broken after moving or renaming the project folder | Virtual environments store absolute paths. Delete `.venv` and repeat Setup steps 2–3 |
| Unpickling errors or `InconsistentVersionWarning` when loading the model | The model was saved with a different scikit-learn version. Retrain with Step 1 |
| `libomp` error when importing xgboost (macOS) | `brew install libomp` |

---

## Known Limitations and Next Steps

- **Optimistic score:** the best model is chosen by its test-set score, so the reported R² is slightly optimistic. Selecting on cross-validation scores, or a separate validation set, would give a fairer estimate.
- **Unfinished components:** `model_evaluations.py` and `model_validation.py` are empty placeholders.
- **Unpinned dependencies:** `requirements.txt` doesn't pin versions. Pin them, for example with `pip freeze`, so results can be reproduced exactly.
- **Development server:** the web app uses Flask's built-in server, which isn't for production. For deployment, use a WSGI server such as `gunicorn app:app`.
- **No styling or tests yet:** the HTML form is unstyled, and the project has no automated tests.
