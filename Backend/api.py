from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Literal

from pathlib import Path
from pathlib import Path
import pickle



app=FastAPI()




# Get the current file directory (Backend folder)
current_dir = Path(__file__).parent

# Construct the path to the model file relative to Backend folder:
# Go up one level (to root), then into ML folder, then model file
model_path = current_dir.parent / 'ML' / 'titanic_model.pkl'

print(f"Looking for model file at: {model_path}")
print(f"Model file exists? {model_path.exists()}")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file not found. Ensure 'titanic_model.pkl' is in the ML folder at project root.")


# Hardcoded encoder mappings based on notebook and standard conventions
ENCODER_MAPPINGS = {
    "Sex": {"female": 0, "male": 1},
    "Embarked": {"C": 0, "S": 2,"NaN":3,"Q": 1}
}

# Define input data schema with string-based categorical fields
class PassengerData(BaseModel):
    Pclass: int
    Sex: Literal["female", "male"]
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: Literal["C", "S","NaN","Q"]

@app.get("/")
async def root():
    return {"message": "Welcome to the Titanic Survival Prediction API. Use /predict to make predictions."}

@app.post("/predict")
async def predict_survival(data: PassengerData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([{
            "Pclass": data.Pclass,
            "Sex": data.Sex,
            "Age": data.Age,
            "SibSp": data.SibSp,
            "Parch": data.Parch,
            "Fare": data.Fare,
            "Embarked": data.Embarked
        }])

        # Encode categorical variables using hardcoded mappings
        input_data["Sex"] = input_data["Sex"].map(ENCODER_MAPPINGS["Sex"])
        input_data["Embarked"] = input_data["Embarked"].map(ENCODER_MAPPINGS["Embarked"])

        # Check for invalid mappings
        if input_data["Sex"].isna().any() or input_data["Embarked"].isna().any():
            raise ValueError("Invalid Sex or Embarked value provided.")

        # Ensure feature order matches training
        feature_order = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        input_data = input_data[feature_order]

        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Interpret LinearRegression output as survival probability
        survival_probability = float(prediction)
        survival = "Yes" if survival_probability >= 0.5 else "No"

        return {
            "survival_probability": survival_probability,
            "survival": survival
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")