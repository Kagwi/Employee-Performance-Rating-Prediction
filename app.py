import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model_random_forest.pkl')

# Configure the Streamlit app
st.set_page_config(page_title="Employee Performance Rating Predictor", layout="wide")
st.title("Employee Performance Rating Prediction")

# Create input sections
with st.form("employee_details"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        EmpEnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        EmpJobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        EmpLastSalaryHikePercent = st.number_input("Salary Hike (%)", min_value=0, max_value=25)
        TotalWorkExperienceInYears = st.number_input("Total Work Experience (Years)", min_value=0)
    
    with col2:
        ExperienceYearsAtThisCompany = st.number_input("Years at Company", min_value=0)
        ExperienceYearsInCurrentRole = st.number_input("Years in Current Role", min_value=0)
        EmpJobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0)
    
    with col3:
        YearsWithCurrManager = st.number_input("Years with Current Manager", min_value=0)
        EmpDepartment = st.selectbox("Department", ["Sales", "R&D", "HR", "Marketing", "Technical"])
        EmpWorkLifeBalance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        EducationBackground = st.selectbox("Education Background", ["Life Sciences", "Medical", "Technical Degree", "Human Resources", "Other"])
        TrainingTimesLastYear = st.number_input("Trainings Last Year", min_value=0)
        Gender = st.selectbox("Gender", ["Male", "Female"])
    
    submit_button = st.form_submit_button("Predict Performance Rating")

# Generate prediction
if submit_button:
    input_data = {
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'EmpJobLevel': EmpJobLevel,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'EmpDepartment': EmpDepartment,
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'BusinessTravelFrequency': BusinessTravelFrequency,
        'EducationBackground': EducationBackground,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'Gender': Gender
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    
    st.subheader("Prediction Result")
    st.metric("Predicted Performance Rating", prediction[0])
