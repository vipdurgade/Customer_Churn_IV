# 🧠 Customer Churn Prediction

This project presents a complete pipeline for predicting customer churn using machine learning techniques. The goal is to help Itzehoer Versicherungen businesses proactively identify customers at risk of leaving and implement retention strategies.

🔗 **Live Demo:** [Streamlit App](https://customerchurnvip.streamlit.app/)

---

## 📂 Project Overview

This ML project predicts whether a customer is likely to churn based on their contract, service usage, and demographic data. It includes data cleaning, exploratory analysis, feature engineering, model building, evaluation, and deployment via Streamlit.

---

## 📊 Dataset

- **Source:** Kaggle / Telco Customer Churn Dataset
- **Rows:** ~7,000
- **Target:** `Churn` (Yes/No)

### 🔑 Key Features Used

| Feature               | Description |
|-----------------------|-------------|
| `gender`              | Customer gender |
| `SeniorCitizen`       | Indicates if the customer is a senior |
| `Partner`, `Dependents` | Family status |
| `tenure`              | Number of months as customer |
| `Contract`            | Type of contract (Month-to-month, etc.) |
| `InternetService`     | Fiber optic, DSL, or None |
| `MonthlyCharges`, `TotalCharges` | Billing information |

---

## 🧼 Data Preprocessing

- Converted `TotalCharges` to numeric
- Encoded categorical variables
- Imputed missing values
- Standardized numerical features

---

## 📈 Exploratory Data Analysis (EDA)

- Visualized churn distribution
- Correlated churn with contract type, tenure, payment method
- Highlighted high-churn customer segments

---

## 🤖 Model Building

- **Models Tried:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - CatBoost (best performance)

- **Selected Model:** CatBoost Classifier

---

## ✅ Evaluation Metrics

| Metric     | Score    |
|------------|----------|
| Accuracy   | ~84%     |
| Precision  | High     |
| Recall     | High     |
| AUC        | Excellent separation ability |

> All metrics were validated using train-test split and `confusion_matrix`, `classification_report`, and ROC curves.

---

## 🚀 Deployment

The final CatBoost model is deployed using Streamlit:

🔗 **[Try the App](https://customerchurnvip.streamlit.app/)**

Features:
- Upload your own CSV file
- Get instant predictions
- View processed input and output

---

## 🗂️ Project Structure



---

## 🔧 How to Run Locally

1. Clone the repo  
   `git clone https://github.com/vipdurgade/Customer_Churn_IV.git`

2. Create a virtual environment  
   `python -m venv venv && source venv/bin/activate` *(or `venv\Scripts\activate` on Windows)*

3. Install dependencies  
   `pip install -r requirements.txt`

4. Run the app  
   `streamlit run app/app.py`

---

## 🧠 Future Improvements

- Add SHAP for model explainability
- Improve performance with hyperparameter tuning
- Connect to real-time customer database/API
- Add login/authentication for business users

---

## 📬 Contact

Made with ❤️ by [Vipul Durgad](https://github.com/vipdurgade)  
For feedback or collaboration: `vipuldurgade@gmail.com`

