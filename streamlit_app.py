
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊", layout="wide")

bundle = joblib.load(Path(__file__).parent / "attrition_model.joblib")

model = bundle["model"]
numerical_cols = bundle["numerical_cols"]
categorical_cols = bundle["categorical_cols"]
num_imputer = bundle["num_imputer"]
cat_imputer = bundle["cat_imputer"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]

st.title("Employee Attrition Prediction")
st.write("Enter employee information to estimate the probability of leaving the company.")

with st.form("employee_form"):
    st.subheader("Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age", 18, 100, 35)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        DistanceFromHome = st.number_input("Distance From Home", 0, 100, 5)

    with col2:
        Education = st.selectbox("Education", [1, 2, 3, 4, 5], index=2)
        EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
        Department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    with col3:
        DailyRate = st.number_input("Daily Rate", 0, 2000, 800)
        HourlyRate = st.number_input("Hourly Rate", 0, 100, 65)
        MonthlyIncome = st.number_input("Monthly Income", 0, 100000, 5000)
        MonthlyRate = st.number_input("Monthly Rate", 0, 30000, 15000)

    st.subheader("Job Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        JobRole = st.selectbox("Job Role", ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Executive", "Sales Representative", "Research Director", "Human Resources"])
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
        NumCompaniesWorked = st.number_input("Previous Companies", 0, 20, 2)
        TotalWorkingYears = st.number_input("Total Working Years", 0, 50, 10)

    with col2:
        YearsAtCompany = st.number_input("Years at Company", 0, 50, 5)
        YearsInCurrentRole = st.number_input("Years in Current Role", 0, 50, 3)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 50, 1)
        YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 50, 3)

    with col3:
        OverTime = st.selectbox("Overtime", ["No", "Yes"])
        PercentSalaryHike = st.number_input("Salary Hike (%)", 0, 100, 12)
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=1)
        TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 3)

    st.subheader("Satisfaction and Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
        JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=3)

    with col2:
        JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)

    with col3:
        PerformanceRating = st.selectbox("Performance Rating", [3, 4])
        WorkLifeBalance = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2)

    submitted = st.form_submit_button("Predict Attrition")

if submitted:
    data = pd.DataFrame([{
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }])

    data[numerical_cols] = num_imputer.transform(data[numerical_cols])
    data[categorical_cols] = cat_imputer.transform(data[categorical_cols])
    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_columns, fill_value=0)
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= 0.5)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Likely to leave")
    else:
        st.success("Prediction: Likely to stay")

    st.metric("Estimated Attrition Probability", f"{probability:.2%}")
    st.progress(probability)
    st.caption("This is a model estimate, not a certainty about an employee.")
