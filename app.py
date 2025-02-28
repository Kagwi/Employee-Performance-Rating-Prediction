import streamlit as st 
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('best_model.pkl')

# Initialize label encoders for categorical variables
label_encoders = {
    'EmpDepartment': LabelEncoder(),
    'BusinessTravelFrequency': LabelEncoder(),
    'EducationBackground': LabelEncoder(),
    'Gender': LabelEncoder()
}

# Fit label encoders with known categories
category_mappings = {
    'EmpDepartment': ['Sales', 'HR', 'Development', 'Data Science', 'R&D', 'Finance'],
    'BusinessTravelFrequency': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'EducationBackground': ['Life Sciences', 'Medical', 'Technical Degree', 'Human Resources', 'Other'],
    'Gender': ['Male', 'Female']
}

for col, classes in category_mappings.items():
    label_encoders[col].fit(classes)

EDUCATION_LEVELS = {
    1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'
}

SATISFACTION_LEVELS = {
    1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
}

WORK_LIFE_BALANCE = {
    1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'
}

PERFORMANCE_RATING = {
    1: ('Low', 'Needs significant improvement'),
    2: ('Good', 'Meets expectations with room for growth'),
    3: ('Excellent', 'Exceeds expectations'),
    4: ('Outstanding', 'Far exceeds expectations')
}

st.set_page_config(page_title="Employee Performance Rating Predictor", layout="wide")
st.title("Employee Performance Rating Prediction")
st.markdown("---")

st.sidebar.image("https://source.unsplash.com/400x300/?business,team", use_column_width=True)
st.sidebar.markdown("### About This App")
st.sidebar.write("This app predicts employee performance ratings based on various factors.")
st.sidebar.write("Adjust the inputs and hit the Predict button to get insights!")

with st.form("employee_details"):
    st.markdown("### Employee Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        EmpEnvironmentSatisfaction = st.selectbox("Environment Satisfaction", list(SATISFACTION_LEVELS.keys()), format_func=lambda x: SATISFACTION_LEVELS[x])
        EmpJobSatisfaction = st.selectbox("Job Satisfaction", list(SATISFACTION_LEVELS.keys()), format_func=lambda x: SATISFACTION_LEVELS[x])
        EmpLastSalaryHikePercent = st.number_input("Salary Hike (%)", min_value=0, max_value=25)
        TotalWorkExperienceInYears = st.number_input("Total Work Experience (Years)", min_value=0)
    
    with col2:
        ExperienceYearsAtThisCompany = st.number_input("Years at Company", min_value=0)
        ExperienceYearsInCurrentRole = st.number_input("Years in Current Role", min_value=0)
        EmpJobLevel = st.selectbox("Job Level", list(EDUCATION_LEVELS.keys()), format_func=lambda x: EDUCATION_LEVELS[x])
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0)
    
    with col3:
        YearsWithCurrManager = st.number_input("Years with Current Manager", min_value=0)
        EmpDepartment = st.selectbox("Department", category_mappings['EmpDepartment'])
        EmpWorkLifeBalance = st.selectbox("Work-Life Balance", list(WORK_LIFE_BALANCE.keys()), format_func=lambda x: WORK_LIFE_BALANCE[x])
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", category_mappings['BusinessTravelFrequency'])
        EducationBackground = st.selectbox("Education Background", category_mappings['EducationBackground'])
        TrainingTimesLastYear = st.number_input("Trainings Last Year", min_value=0)
        Gender = st.selectbox("Gender", category_mappings['Gender'])
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
        'EmpDepartment': label_encoders['EmpDepartment'].transform([EmpDepartment])[0],
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'BusinessTravelFrequency': label_encoders['BusinessTravelFrequency'].transform([BusinessTravelFrequency])[0],
        'EducationBackground': label_encoders['EducationBackground'].transform([EducationBackground])[0],
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'Gender': label_encoders['Gender'].transform([Gender])[0],
        'Attrition': 1 if Attrition == "Yes" else 0
    }
    
    input_df = pd.DataFrame([input_data])

    if hasattr(model, 'feature_names_in_'):
        input_df = input_df[model.feature_names_in_]
    
    prediction = model.predict(input_df)[0]
    rating_text, rating_description = PERFORMANCE_RATING.get(prediction, (prediction, "Unknown"))
    
    st.subheader("Prediction Result")
    st.metric("Predicted Performance Rating", f"{rating_text} ({prediction})")
    st.write(f"**What this means:** {rating_description}")
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({"Feature": model.feature_names_in_, "Importance": model.feature_importances_})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(y=importance_df["Feature"], x=importance_df["Importance"], palette="viridis", ax=ax)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance for Performance Prediction")
        st.pyplot(fig)
