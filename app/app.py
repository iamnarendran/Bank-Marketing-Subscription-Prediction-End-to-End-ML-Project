from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model pipeline
try:
    model_pipeline = joblib.load("lgb_bank_marketing_model.joblib")
except FileNotFoundError:
    print("Error: 'bank_marketing_model.joblib' not found. Please make sure the model file exists.")
    model_pipeline = None # Set model_pipeline to None to avoid further errors

app = FastAPI()

# Define the input data model using Pydantic
class BankMarketingInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


# Define the prediction endpoint
@app.post("/predict")
def predict(data: BankMarketingInput):
    if model_pipeline is None:
        return {"error": "Model not loaded."}

    # Convert the input data to a pandas DataFrame
    df_input = pd.DataFrame([data.model_dump()])

    # Make prediction
    prediction_proba = model_pipeline.predict_proba(df_input)[:, 1]
    # You can choose a threshold here based on your evaluation results
    threshold = 0.6
    prediction = int(prediction_proba[0] >= threshold)

    return {"probability": prediction_proba[0], "prediction": prediction}

# To run this FastAPI app in Colab, you would typically use something like uvicorn:
# !uvicorn main:app --reload --port 8000 &
# Replace 'main' with the name of your Python file if running outside Colab.
# In Colab, you might need to expose the port using ngrok or a similar service.
