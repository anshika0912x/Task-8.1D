# SIT307 Task 8.1P – Housing Price Prediction

This project develops a machine learning model to estimate residential property sale prices using historical sales data from three Sydney suburbs:

- Mosman (2088)

- Marrickville (2204)

- Blacktown (2148)

The project includes data cleaning, exploratory data analysis, feature engineering, regression modelling, model evaluation, feature importance analysis, prediction error investigation, and deployment using Streamlit.

## Project Objective

The objective of this project is to predict property sale prices using features such as:

- Suburb

- Property type

- Bedrooms

- Bathrooms

- Parking spaces

- Property area

- Sale month

Three regression approaches were evaluated:

1. Multiple Linear Regression

2. Log-Target Linear Regression

3. Polynomial Regression

The **Log-Target Linear Regression** model produced the strongest test performance, with approximately:

- MAE: $417,310

- RMSE: $907,004

- R²: 0.770

## Repository contents

| File | Description |
|---|---|
| `Housing.csv` | Original manually collected property dataset (108 properties) |
| `Housing_Cleaned.csv` | Cleaned dataset used for analysis (106 properties) |
| `8.1D.ipynb` | Full notebook — data preparation, EDA, feature engineering, model development, evaluation, and prediction-failure analysis |
| `8.1D.py` | Streamlit deployment application |
| `log_linear_model.pkl` | Trained log-target linear regression pipeline used by the app |
| `requirements.txt` | Python packages required to run the notebook and app |

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/anshika0912x/Task-8.1D.git
   cd Task-8.1D
   ```
2. (Recommended) create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Running the notebook

Open `8.1D.ipynb` in Jupyter or VS Code and run all cells top to bottom. This reproduces the
full analysis and, in the final cells, saves the trained model as `log_linear_model.pkl` in the
same folder.

## Running the deployment app

Make sure `log_linear_model.pkl` has been generated (see above) and sits in the same folder as
`8.1D.py`, then run:

```bash
python3 -m streamlit run 8.1D.py
```

This opens the app at `http://localhost:8501`. Enter a suburb, property type, bedrooms,
bathrooms, parking spaces, area, and sale month to get an estimated sale price.

**Note:** if you see a `ModuleNotFoundError: No module named 'sklearn'` when launching, it
usually means `streamlit` and `python3` are resolving to different Python installations on your
machine. Running the app with `python3 -m streamlit run 8.1D.py` (as above), rather than the
bare `streamlit run` command, ensures both use the same interpreter that `requirements.txt` was
installed into.

## Model summary

Three regression approaches were compared on an 80/20 train/test split (stratified by suburb):

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Multiple Linear Regression | $670,673 | $1,111,803 | 0.654 |
| **Log-Target Linear Regression (selected)** | **$417,310** | **$907,004** | **0.770** |
| Polynomial Regression (deg. 2) | $843,628 | $1,307,346 | 0.521 |

The log-target linear regression was selected as the final model deployed in the app, based on
the best test-set generalisation and the smallest train/test R² gap.
