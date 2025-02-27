import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessing artifacts
model = joblib.load('best_model_random_forest.pkl')  # Make sure this is the actual model object
scaler = joblib.load('scaler.pkl')  # Only include if you actually used scaling in training

# Create categorical mappings (must match what was used in training)
CATEGORICAL_MAPPINGS = {
    'EmpDepartment': {
        'Sales': 0, 'R&D': 1, 'HR': 2, 'Marketing': 3, 'Technical': 4
    },
    'BusinessTravelFrequency': {
        'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2
    },
    'EducationBackground': {
        'Life Sciences': 0, 'Medical': 1, 'Technical Degree': 2, 
        'Human Resources': 3, 'Other': 4
    },
    'Gender': {'Male': 0, 'Female': 1}
}

st.set_page_config(page_title="Employee Performance Rating Predictor", layout="wide")
st.title("Employee Performance Rating Prediction")

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
        EmpDepartment = st.selectbox("Department", list(CATEGORICAL_MAPPINGS['EmpDepartment'].keys()))
        EmpWorkLifeBalance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", 
                                            list(CATEGORICAL_MAPPINGS['BusinessTravelFrequency'].keys()))
        EducationBackground = st.selectbox("Education Background", 
                                        list(CATEGORICAL_MAPPINGS['EducationBackground'].keys()))
        TrainingTimesLastYear = st.number_input("Trainings Last Year", min_value=0)
        Gender = st.selectbox("Gender", list(CATEGORICAL_MAPPINGS['Gender'].keys()))
    
    submit_button = st.form_submit_button("Predict Performance Rating")

if submit_button:
    # Convert categorical features to numerical using mappings
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
        'EmpDepartment': CATEGORICAL_MAPPINGS['EmpDepartment'][EmpDepartment],
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'BusinessTravelFrequency': CATEGORICAL_MAPPINGS['BusinessTravelFrequency'][BusinessTravelFrequency],
        'EducationBackground': CATEGORICAL_MAPPINGS['EducationBackground'][EducationBackground],
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'Gender': CATEGORICAL_MAPPINGS['Gender'][Gender]
    }
    
    # Create DataFrame and ensure column order matches training data
    input_df = pd.DataFrame([input_data])
    
    # Align columns with expected features (modified to handle older scikit-learn versions)
    try:
        # For scikit-learn >= 1.0
        input_df = input_df[model.feature_names_in_]
    except AttributeError:
        # For older versions, use the original feature order
        input_df = input_df[['EmpEnvironmentSatisfaction', 'EmpJobSatisfaction', 'EmpLastSalaryHikePercent',
                            'TotalWorkExperienceInYears', 'ExperienceYearsAtThisCompany',
                            'ExperienceYearsInCurrentRole', 'EmpJobLevel', 'YearsSinceLastPromotion',
                            'YearsWithCurrManager', 'EmpDepartment', 'EmpWorkLifeBalance',
                            'BusinessTravelFrequency', 'EducationBackground', 'TrainingTimesLastYear',
                            'Gender']]
    
    # Apply scaling if used in training
    if 'scaler' in locals():
        scaled_input = scaler.transform(input_df)
    else:
        scaled_input = input_df
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Display results
    st.subheader("Prediction Result")
    st.metric("Predicted Performance Rating", prediction[0])
