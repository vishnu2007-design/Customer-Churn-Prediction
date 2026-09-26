# Customer Churn Prediction

## 📌 Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to churn (leave) or stay based on their demographic, service, and account-related information.

The project uses the IBM Telco Customer Churn dataset and includes the complete machine learning workflow from data preprocessing to a Flask-based web application.

## 🎯 Objective

The main objective of this project is to:

- Predict customer churn
- Identify customers who may be at risk of leaving
- Compare different classification algorithms
- Evaluate model performance using multiple metrics
- Provide predictions through a web application

## 🔄 Project Workflow

```text
Data
  ↓
Data Cleaning
  ↓
Exploratory Data Analysis (EDA)
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Feature Selection
  ↓
Train/Test Split
  ↓
One-Hot Encoding
  ↓
Model Comparison
  ↓
Model Evaluation
  ↓
Hyperparameter Tuning
  ↓
Final Model
  ↓
Flask Web Application
```

## 🧹 Data Preprocessing

- Removed duplicate records
- Removed the `customerID` column
- Converted `TotalCharges` into numerical format
- Handled missing values
- Separated features and target
- Used an 80/20 stratified train-test split
- Applied One-Hot Encoding to categorical features

## ⚙️ Feature Engineering

### TotalServices

Counts the number of subscribed services among:

- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

### TenureGroup

Customers were grouped based on their tenure:

- New: 0–12 months
- Short-term: 13–24 months
- Medium-term: 25–48 months
- Long-term: 49+ months

`AverageMonthlySpend` was initially created but removed during feature selection because it was highly correlated with `MonthlyCharges`.

## 🤖 Machine Learning Models

Five classification algorithms were compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🏆 Final Model

After model comparison and hyperparameter tuning using `GridSearchCV`, Logistic Regression with `C = 0.1` was selected as the final model.

### Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 80.50% |
| Precision | 67.75% |
| Recall | 50.27% |
| F1-Score | 57.72% |
| ROC-AUC | 84.09% |

## 🌐 Flask Web Application

The trained model was integrated into a Flask web application.

The application allows users to enter customer information such as:

- Tenure
- Contract type
- Internet service
- Monthly charges
- Total charges
- Payment method
- Technical support
- Online security
- Streaming services
- Other customer details

The application provides:

- Churn prediction
- Predicted churn probability

### Example

```text
Prediction:
Customer is likely to churn.

Predicted churn probability:
71.61%
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS
- Git
- GitHub

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── churn_preprocessor.pkl
├── final_churn_model.pkl
├── requirements.txt
├── Telco-Customer-Churn.csv
└── .gitignore
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/vishnu2007-design/Customer-Churn-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Customer-Churn-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open the application

```text
http://127.0.0.1:5000
```

## 💡 Real-World Applications

- Telecommunications
- Internet service providers
- Subscription services
- Online platforms
- Financial services
- Membership-based businesses

## 📈 Future Improvements

- Deploying the application to a cloud platform
- Adding probability-based risk categories
- Improving model performance through additional tuning
- Adding interactive analytics dashboards
- Implementing automated model retraining

## 👨‍💻 Author

**Jaini Vishnu**

Machine Learning | Python | Data Science
