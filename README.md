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

## Repository Files

- `Housing.csv`  
  Original property dataset collected for the project.

- `Housing_Cleaned.csv`  
  Cleaned dataset used for analysis and modelling.

- `8_1D_extended.ipynb`  
  Main Jupyter Notebook containing data preparation, exploratory analysis, feature engineering, model development, evaluation, and interpretation.

- `8.1D.py`  
  Streamlit application for entering property characteristics and generating a predicted sale price.

- `log_linear_model.pkl`  
  Saved trained log-target linear regression model used by the Streamlit application.

- `requirements.txt`  
  Python packages required to run the project.

## Installation

Clone or download this repository, then install the required Python packages:

```bash
pip install -r requirements.txt
