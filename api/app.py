
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Employee Attrition Prediction API")

bundle = joblib.load("../attrition_model.joblib")
model = bundle["model"]
numerical_cols = bundle["numerical_cols"]
categorical_cols = bundle["categorical_cols"]
num_imputer = bundle["num_imputer"]
cat_imputer = bundle["cat_imputer"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]

class Employee(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

@app.get("/")
def home():
    return {"message": "Employee Attrition Prediction API is running"}

@app.post("/predict")
def predict(employee: Employee):
    data = pd.DataFrame([employee.model_dump()])
    data[numerical_cols] = num_imputer.transform(data[numerical_cols])
    data[categorical_cols] = cat_imputer.transform(data[categorical_cols])
    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_columns, fill_value=0)
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "result": "Likely to leave" if prediction == 1 else "Likely to stay",
        "attrition_probability": round(probability, 4)
    }