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
```bash
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
```

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
- **LightGBM** gave best validation results:
- Test Set Evaluation (Threshold = 0.6 ):
- Accuracy: 0.8729567891244538
- Precision: 0.4527098831030818
- Recall: 0.6120689655172413
- F1: 0.5204642638973732
- AUC: 0.8015438523670866
- Hyperparameter tuning via **RandomizedSearchCV**
- Optimal threshold determined for best Recall-F1 balance

---
## 📈 Business goals based on Threshold

| Goal                                       | Recommended Threshold | Reason                            |
| ------------------------------------------ | --------------------- | --------------------------------- |
|**Maximize Recall (find all subscribers)** | 0.40–0.45             | catches ~70% yes clients          |
| **Balanced F1 (marketing balance)**        | 0.50–0.60             | F1 ≈ 0.49–0.52 → best overall mix |
| **Maximize Precision (only best leads)**   | 0.70                  | fewer false positives             |
---

## 🧮 Final Test Performance
| **Metric**    | **Score** |
| --------- | --------- |
| **Accuracy**  | **0.873** |
| **Precision** | **0.453** |
| **Recall**    | **0.612** |
| **F1**        | **0.520** |
| **AUC**       | **0.801** |

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
```bash

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
```
Response:

{"prediction": 1, "probability": 0.76}

-----
## 🐳 Docker setup

1. Build docker image
```bash
 docker build -t bank-subscriber-api .
```
2. Run container
```bash
docker run -p 8000:8000 bank-subscriber-api
```
-----

## Learnings & Key Takeaways

- Understood end-to-end ML lifecycle (EDA → Deployment)

- Learned threshold tuning to handle class imbalance

- Practiced explainable AI (SHAP) for model transparency

- Explored FastAPI deployment for productionization

- Gained experience writing Dockerfiles and managing environments


