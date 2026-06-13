# Bank Deposit Subscription Prediction

Machine Learning project focused on predicting whether a bank customer will subscribe to a term deposit product.

## Project Overview

The objective of this project is to build and evaluate machine learning models capable of predicting customer subscription to a term deposit campaign using demographic, financial and marketing-related information.

The project includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing and feature engineering
* Model training and evaluation
* Competition predictions
* Interactive Streamlit application for real-time predictions

## Dataset

The dataset contains information about bank customers, including:

* Demographic attributes (age, job, marital status, education)
* Financial information (balance, loans, default status)
* Marketing campaign information
* Previous contact history

Target variable:

* `deposit`: Indicates whether the customer subscribed to a term deposit product.

## Project Structure

```text
bank-deposit-subscription-prediction/

├── data/
│   ├── raw/
│   └── predictions/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── predicciones_competicion.ipynb
│
├── models/
│   └── modelo_final.joblib
│
├── app/
│   └── mystreamlit.py
│
├── images/
│
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis

* Data quality assessment
* Missing values analysis
* Class balance analysis
* Variable distribution analysis
* High-cardinality categorical variables inspection

### 2. Data Preprocessing

* Missing value handling
* Feature engineering
* Categorical variable encoding
* Numerical feature preparation

### 3. Model Development

Several machine learning models were evaluated and compared to identify the best-performing solution.

The final model was exported and stored using Joblib for deployment.

### 4. Deployment

A Streamlit web application was developed to allow users to enter customer information and obtain subscription predictions in real time.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* Jupyter Notebook

## Results

Main evaluation metrics:

* Accuracy: XX%
* Precision: XX%
* Recall: XX%
* F1 Score: XX%
* ROC-AUC: XX%

## Streamlit Application

The project includes an interactive Streamlit application that allows users to:

* Enter customer information
* Generate predictions instantly
* Visualize whether the customer is likely to subscribe to a term deposit

## Future Improvements

* Hyperparameter optimization
* Advanced feature engineering
* Explainable AI techniques (SHAP)
* Model monitoring and retraining pipeline

