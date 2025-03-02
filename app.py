import streamlit as st 
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load('best_model.pkl')

# Load pre-trained label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Define mappings for categorical features
JOB_LEVELS = {
    1: 'Entry Level', 2: 'Mid Level', 3: 'Senior Level', 4: 'Executive Level'
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

st.sidebar.markdown("### About This App")
st.sidebar.write("This app predicts employee performance ratings based on various factors.")
st.sidebar.write("Adjust the inputs and hit the Predict button to get insights!")

# Visually appealing explanation section
st.markdown("## How to Use the Application")
st.markdown("Use the following guide to understand input values:")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Job Level")
    st.markdown("- 1️⃣ Entry Level")
    st.markdown("- 2️⃣ Mid Level")
    st.markdown("- 3️⃣ Senior Level")
    st.markdown("- 4️⃣ Executive Level")

    st.markdown("### Job & Environment Satisfaction")
    st.markdown("- 1️⃣ Low")
    st.markdown("- 2️⃣ Medium")
    st.markdown("- 3️⃣ High")
    st.markdown("- 4️⃣ Very High")

with col2:
    st.markdown("### Work-Life Balance")
    st.markdown("- 1️⃣ Bad")
    st.markdown("- 2️⃣ Good")
    st.markdown("- 3️⃣ Better")
    st.markdown("- 4️⃣ Best")
    
    st.markdown("### Performance Rating Explanation")
    for key, value in PERFORMANCE_RATING.items():
        st.markdown(f"- {key}️⃣ {value[0]}: *{value[1]}*")

st.markdown("---")

# Form for user input
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
        EmpJobLevel = st.selectbox("Job Level", list(JOB_LEVELS.keys()), format_func=lambda x: JOB_LEVELS[x])
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0)
    
    with col3:
        YearsWithCurrManager = st.number_input("Years with Current Manager", min_value=0)
        EmpDepartment = st.selectbox("Department", label_encoders['EmpDepartment'].classes_)
        EmpWorkLifeBalance = st.selectbox("Work-Life Balance", list(WORK_LIFE_BALANCE.keys()), format_func=lambda x: WORK_LIFE_BALANCE[x])
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", label_encoders['BusinessTravelFrequency'].classes_)
        EducationBackground = st.selectbox("Education Background", label_encoders['EducationBackground'].classes_)
        TrainingTimesLastYear = st.number_input("Trainings Last Year", min_value=0)
        Gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
        Attrition = st.selectbox("Attrition Status", ["No", "Yes"])
    
    submit_button = st.form_submit_button("Predict Performance Rating")

if submit_button:
    input_data = {
        "EmpEnvironmentSatisfaction": EmpEnvironmentSatisfaction,
        "EmpJobSatisfaction": EmpJobSatisfaction,
        "EmpLastSalaryHikePercent": EmpLastSalaryHikePercent,
        "TotalWorkExperienceInYears": TotalWorkExperienceInYears,
        "ExperienceYearsAtThisCompany": ExperienceYearsAtThisCompany,
        "ExperienceYearsInCurrentRole": ExperienceYearsInCurrentRole,
        "EmpJobLevel": EmpJobLevel,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "EmpDepartment": label_encoders['EmpDepartment'].transform([EmpDepartment])[0],
        "EmpWorkLifeBalance": EmpWorkLifeBalance,
        "Attrition": 1 if Attrition == "Yes" else 0,
        "BusinessTravelFrequency": label_encoders['BusinessTravelFrequency'].transform([BusinessTravelFrequency])[0],
        "EducationBackground": label_encoders['EducationBackground'].transform([EducationBackground])[0],
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "Gender": label_encoders['Gender'].transform([Gender])[0]
    }
    
    input_df = pd.DataFrame([input_data])

    feature_order = [
        "EmpEnvironmentSatisfaction", "EmpJobSatisfaction", "EmpLastSalaryHikePercent", 
        "TotalWorkExperienceInYears", "ExperienceYearsAtThisCompany", "ExperienceYearsInCurrentRole", 
        "EmpJobLevel", "YearsSinceLastPromotion", "YearsWithCurrManager", "EmpDepartment", 
        "EmpWorkLifeBalance", "Attrition", "BusinessTravelFrequency", "EducationBackground", 
        "TrainingTimesLastYear", "Gender"
    ]
    
    input_df = input_df[feature_order]

    prediction = model.predict(input_df)[0]
    rating_text, rating_description = PERFORMANCE_RATING.get(prediction, (prediction, "Unknown"))

    st.subheader("Prediction Result")
    st.metric("Predicted Performance Rating", f"{rating_text} ({prediction})")
    st.write(f"**What this means:** {rating_description}")
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({"Feature": feature_order, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(y=importance_df["Feature"], x=importance_df["Importance"], palette="viridis", ax=ax)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance for Performance Prediction")
        st.pyplot(fig)
