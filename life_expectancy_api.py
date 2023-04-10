import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel

# FastAPI libray
from fastapi import FastAPI

# Initiate app instance
app = FastAPI(title='Life Expectancy Prediction')

regressor = joblib.load('models/life_expectancy_model.joblib')

class Data(BaseModel):
    Country: str,
    Infant_deaths: float,
    Under_five_deaths: float,
    Adult_mortality: float,
    Alcohol_consumption: float,
    Hepatitis_B: int,
    Measles: int,
    BMI: float,
    Polio:int,
    Diphtheria:int,
    Incidents_HIV:float,
    GDP_per_capita:int,
    Population_mln:float,
    Thinness_ten_nineteen_years:float,
    Thinness_five_nine_years:float,
    Schooling:float,
    Economy_status_Developed:int,
    Economy_status_Developing:int
      
      

      
@app.post("/predict")
def predict(data: Data):
    # Extract data in correct order
    data_dict = data.dict()
    data_df = pd.DataFrame.from_dict([data_dict])
    # Select features required for making prediction
    #data_df = data_df[features]
    # Perform label encoding for categorical features
    #data_df[categorical_features] = le.transform(data_df[categorical_features])
    # Create prediction
    prediction = regressor.predict(data_df)
    # Map prediction to appropriate label
    #prediction_label = ['Placed' if label == 0 else 'Not Placed' for label in prediction ]
    # Return response back to client
    return {"prediction": prediction[0]}

if __name__ == '__main__':
    uvicorn.run("life_expectancy_api:app", host="0.0.0.0", port=8000, reload=True)
