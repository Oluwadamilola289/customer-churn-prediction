# 📊 Customer Churn Prediction System

An end-to-end machine learning application that predicts whether a telecom customer is likely to churn based on demographic, billing, and service usage information.

Built using Python, Scikit-Learn, Pandas, and Streamlit.

---

## 🚀 Live Demo

Coming Soon

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses.

This project builds a complete machine learning pipeline that:

- Cleans raw customer data
- Performs feature engineering
- Trains multiple ML models
- Selects the best-performing model
- Predicts customer churn
- Displays predictions through an interactive Streamlit dashboard

---

## ✨ Features

- Customer churn prediction
- Interactive Streamlit dashboard
- Data preprocessing pipeline
- Feature engineering
- Multiple model comparison
- Prediction confidence score
- Visual model comparison chart
- Production-ready pipeline

---

## 📂 Project Structure

```
customer-churn-prediction/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── dashboard.py
├── README.md
└── requirements.txt
```

---

## 🧠 Machine Learning Pipeline

1. Data Cleaning
2. Feature Engineering
3. Train/Test Split
4. Model Training
5. Model Evaluation
6. Model Comparison
7. Prediction Dashboard

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **81.6%** |
| Gradient Boosting | 81.1% |
| Random Forest | 79.5% |
| Decision Tree | 72.5% |

Best Model:

**Logistic Regression**

---

## 📷 Dashboard

### Main Dashboard

![Dashboard](images/dashboard.png)

---

### Customer Prediction

![Prediction](images/prediction_stay.png)

---

### High Risk Customer

![Prediction](images/prediction_churn.png)

---

### Model Comparison

![Models](images/model_comparison.png)

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Joblib

---

## 📊 Dataset

Telco Customer Churn Dataset

- 7,043 Customers
- 20 Features
- Binary Classification

Target Variable:

```
Churn
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Oluwadamilola289/customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run dashboard.py
```

---

## 🔮 Future Improvements

- XGBoost implementation
- Hyperparameter tuning
- SHAP explainability
- Docker deployment
- Cloud deployment
- REST API

---

## 👩🏽‍💻 Author

**Oluwadamilola Osho**

AI Engineer | Data Analyst | Machine Learning Enthusiast

LinkedIn:
https://www.linkedin.com/in/damilola-e-osho



GitHub:
https://github.com/Oluwadamilola289