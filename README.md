# 🚗 Sales Prediction
---
## 📌 Overview

This machine learning project predicts a customer’s car purchase amount using demographic and financial features such as age, annual salary, credit card debt, and net worth.

✅ Features:
- Trained regression models
- Interactive **Streamlit web app**
- Clean and modular Python code
- EDA and model development in Jupyter Notebook
- Prediction CSV generation for business insights

---

## 📂 [Dataset](https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction)

The dataset includes 500 rows and the following fields:

- `Customer Name`
- `Email`
- `Country`
- `Gender`
- `Age`
- `Annual Salary`
- `Credit Card Debt`
- `Net Worth`
- `Car Purchase Amount` (Target)

---

## 🧹 Data Preprocessing

Performed in the notebook and script:

- Handling missing/null values
- Outlier detection and removal
- Categorical encoding
- Feature scaling with `StandardScaler`

---

## 🤖 Machine Learning Models

- Linear Regression (default)
- Random Forest Regressor (optional extension)

Models are trained using `sklearn`, and evaluated on test data using:

- R² Score
- RMSE (Root Mean Squared Error)

---

## 📊 Evaluation Metrics

These metrics help assess model performance:

- **R² Score**: Measures the proportion of variance explained
- **RMSE**: Indicates average prediction error

---

## 💻 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sales-prediction.git
cd sales-prediction

pip install -r requirements.txt
```
### 2. Run app2.ipynb 
```bash
jupyter notebook app2.ipynb
```
### 3. Launch the streamlit App
```bash
streamlit run app.py
```
---
### Project Structure
 sales-prediction/
├── app.py                   # Streamlit web app
├── app2.ipynb               # EDA  + modeling in Jupyter Notebook
├── main.py                  # Optional: runs model pipeline script
├── data/
│   └── car_purchase.csv     # Raw dataset
├── models/
│   └── model.pkl            # Saved ML model
├── outputs/
│   └── predictions.csv      # Output predictions
├── src/
│   ├── modeling.py          # Model training logic
│   └── utils.py             # Utility functions (save_predictions)
├── requirements.txt         # Required Python packages
└── README.md                # This file

---
# core components

1. app.py
Streamlit application for interactive predictions
```bash
streamlit run app.py
```
It allows:

    Uploading data

    Viewing predictions

    Model result summaries
---    

2. app2.ipynb

Comprehensive notebook with:

    Exploratory Data Analysis (EDA)

    Model training and validation

    Visualization of results
---
3. src/utils.py
 saving predictions and loading data 

 ---
 # Output
 1. Predictions are saved in the `outputs/predictions.csv` file for further analysis and reporting.

 2. You can interactively explore predictions through app.py
