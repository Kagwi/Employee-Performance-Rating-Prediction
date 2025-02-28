import streamlit as st 
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler

# Categorical mappings (must match what was used during training)
CATEGORICAL_MAPPINGS = {
    'EmpDepartment': {'Sales': 0, 'HR': 1, 'Development': 2, 'Data Science': 3, 'R&D': 4, 'Finance': 5},
    'BusinessTravelFrequency': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
    'EducationBackground': {'Life Sciences': 0, 'Medical': 1, 'Technical Degree': 2, 'Human Resources': 3, 'Other': 4},
    'Gender': {'Male': 0, 'Female': 1}
}

NUMERICAL_FEATURES = [
    "EmpEnvironmentSatisfaction", "EmpJobSatisfaction", "EmpLastSalaryHikePercent",
    "TotalWorkExperienceInYears", "ExperienceYearsAtThisCompany", "ExperienceYearsInCurrentRole",
    "EmpJobLevel", "YearsSinceLastPromotion", "YearsWithCurrManager", "TrainingTimesLastYear"
]

st.set_page_config(page_title="Employee Performance Rating Predictor", layout="wide")
st.title("Employee Performance Rating Prediction")
st.markdown("---")

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
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", list(CATEGORICAL_MAPPINGS['BusinessTravelFrequency'].keys()))
        EducationBackground = st.selectbox("Education Background", list(CATEGORICAL_MAPPINGS['EducationBackground'].keys()))
        TrainingTimesLastYear = st.number_input("Trainings Last Year", min_value=0)
        Gender = st.selectbox("Gender", list(CATEGORICAL_MAPPINGS['Gender'].keys()))
        Attrition = st.selectbox("Attrition Status", ["No", "Yes"])

    submit_button = st.form_submit_button("Predict Performance Rating")

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
        'EmpDepartment': CATEGORICAL_MAPPINGS['EmpDepartment'][EmpDepartment],
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'BusinessTravelFrequency': CATEGORICAL_MAPPINGS['BusinessTravelFrequency'][BusinessTravelFrequency],
        'EducationBackground': CATEGORICAL_MAPPINGS['EducationBackground'][EducationBackground],
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'Gender': CATEGORICAL_MAPPINGS['Gender'][Gender],
        'Attrition': 1 if Attrition == "Yes" else 0
    }
    
    input_df = pd.DataFrame([input_data])

    # Ensure input_df matches model feature order
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df[model.feature_names_in_]

    # Scale numerical features
    input_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])

    # Predict using the model
    prediction = model.predict(input_df)[0]

    PERFORMANCE_RATING = {
        1: ('Low', 'Needs significant improvement'),
        2: ('Good', 'Meets expectations with room for growth'),
        3: ('Excellent', 'Exceeds expectations'),
        4: ('Outstanding', 'Far exceeds expectations')
    }

    rating_text, rating_description = PERFORMANCE_RATING.get(prediction, (prediction, "Unknown"))

    st.subheader("Prediction Result")
    st.metric("Predicted Performance Rating", f"{rating_text} ({prediction})")
    st.write(f"**What this means:** {rating_description}")

    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({"Feature": model.feature_names_in_, "Importance": model.feature_importances_})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        
        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance for Performance Prediction")
        st.pyplot(fig)
