# 🏦 Bank Marketing Subscription Prediction  
_End-to-End Machine Learning Project | FastAPI + LightGBM + SHAP + Docker_

## 📘 Project Overview
This project predicts whether a bank client will subscribe to a term deposit (`y = yes/no`) based on their personal and campaign-related data.  
The goal is to help marketing teams **target potential customers efficiently** while understanding the **drivers behind each prediction** using explainable AI (SHAP).

---

## 🎯 Objectives
- Perform complete **EDA and Feature Engineering** on the UCI Bank Marketing dataset  
- Build a **robust ML model** (LightGBM) with hyperparameter tuning  
- Implement **Threshold Tuning** for better recall on imbalanced data  
- Provide **Model Explainability** using SHAP visualizations  
- Deploy a **FastAPI endpoint** to serve predictions  
- (Optional) Containerize the app with a **Dockerfile** for production readiness

---

## 🧠 Tech Stack
| Layer | Tools / Libraries |
|-------|-------------------|
| **Data Analysis** | Pandas, NumPy, Matplotlib, Seaborn |
| **Modeling** | Scikit-learn, LightGBM, RandomizedSearchCV |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Deployment** | FastAPI, Uvicorn |
| **Containerization (Optional)** | Docker |
| **Environment** | Python 3.11, Ubuntu/Linux |

---

## 📂 Folder Structure
bank-subscription-project/
│
├── app.py # FastAPI app for prediction endpoint
├── model_pipeline.pkl # Trained LightGBM model pipeline
├── requirements.txt # Dependencies list
├── Dockerfile # Docker setup (optional)
├── data/
│ ├── bank-additional-full.csv
│ └── bank-additional.csv
├── notebooks/
│ └── EDA_and_Modeling.ipynb
└── README.md


---

## 🔍 Data Description
Dataset: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

| Feature | Description |
|----------|-------------|
| age | Client’s age |
| job | Type of job |
| marital | Marital status |
| education | Education level |
| default | Has credit in default? |
| housing | Has housing loan? |
| loan | Has personal loan? |
| contact | Communication type |
| month | Last contact month |
| day_of_week | Last contact day |
| duration | Last contact duration (removed before modeling) |
| campaign | Contacts during campaign |
| pdays | Days since last contact |
| previous | Number of previous contacts |
| poutcome | Outcome of previous campaign |
| emp.var.rate | Employment variation rate |
| cons.price.idx | Consumer price index |
| cons.conf.idx | Consumer confidence index |
| euribor3m | 3-month Euribor rate |
| nr.employed | Number of employees |
| y | Target: client subscribed (1=yes, 0=no) |

---

## 🧩 Feature Engineering Steps
- Replaced “unknown” values using mode imputation and domain logic  
- Transformed `pdays = 999` → `-1` to indicate “no previous contact”  
- Detected and handled **rare categories** in categorical columns  
- Used **Ordinal + One-Hot Encoding** via Scikit-learn `ColumnTransformer`  
- Standardized continuous features  
- Split into `train`, `validation`, `test` (70/15/15 with stratification)

---

## 🤖 Model Training & Tuning
- Tried **Logistic Regression (L1/L2)**, **XGBoost**, **CatBoost**, **LightGBM**, **Random Forest**  
- **LightGBM** gave best validation results:
Accuracy: 0.9115
Precision: 0.5941
Recall: 0.6952
F1-Score: 0.6407
AUC: 0.9474

- Hyperparameter tuning via **RandomizedSearchCV**
- Optimal threshold determined for best Recall-F1 balance

---

## 🧮 Final Test Performance
| Metric | Score |
|--------|--------|
| Accuracy | 0.84 |
| Precision | 0.38 |
| Recall | 0.66 |
| F1-Score | 0.49 |
| AUC | 0.80 |
| Threshold | 0.5 |

---

## 📊 Model Explainability (SHAP)
Used SHAP to interpret feature contributions to predictions.

### 🔹 Global Explainability
`shap.summary_plot(shap_values, X_val_prep_df)`
- Features like `previous`, `euribor3m`, and `housing` had highest influence.

### 🔹 Local Explainability
`shap.force_plot(...)`
- Explains *why* a specific client was predicted to subscribe or not.

---

## ⚡ FastAPI Deployment

### Run locally:
```bash
uvicorn app:app --reload
```

## Sample API request:

POST /predict

{
  "age": 45,
  "job": "technician",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp.var.rate": 1.4,
  "cons.price.idx": 93.2,
  "cons.conf.idx": -36.4,
  "euribor3m": 4.85,
  "nr.employed": 5191.0
}

Response:

{"prediction": 1, "probability": 0.76}

-----

## Learnings & Key Takeaways

- Understood end-to-end ML lifecycle (EDA → Deployment)

- Learned threshold tuning to handle class imbalance

- Practiced explainable AI (SHAP) for model transparency

- Explored FastAPI deployment for productionization

- Gained experience writing Dockerfiles and managing environments


