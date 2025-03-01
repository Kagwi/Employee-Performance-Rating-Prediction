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

st.markdown("## 📌 How to Use the Application")
st.write("To ensure accurate predictions, please follow these guidelines while entering information:")

st.markdown("### 🏢 Job-Related Inputs")
st.markdown("- **Job Level**: Select the employee's job level:")
st.write("  - 1️⃣ Entry Level")
st.write("  - 2️⃣ Mid Level")
st.write("  - 3️⃣ Senior Level")
st.write("  - 4️⃣ Executive Level")

st.markdown("### 😊 Satisfaction Levels")
st.markdown("- **Environment Satisfaction** and **Job Satisfaction** ratings are categorized as follows:")
st.write("  - 1️⃣ Low")
st.write("  - 2️⃣ Medium")
st.write("  - 3️⃣ High")
st.write("  - 4️⃣ Very High")

st.markdown("### ⚖️ Work-Life Balance")
st.write("- Select an appropriate rating for the work-life balance:")
st.write("  - 1️⃣ Bad")
st.write("  - 2️⃣ Good")
st.write("  - 3️⃣ Better")
st.write("  - 4️⃣ Best")

st.markdown("### 💰 Salary & Experience")
st.write("- **Salary Hike (%)**: Enter the percentage increase in salary (0-25%).")
st.write("- **Total Work Experience (Years)**: Enter the employee’s total years of work experience.")
st.write("- **Years at Company**: Enter the number of years the employee has worked in this company.")
st.write("- **Years in Current Role**: Enter how many years the employee has been in their current role.")
st.write("- **Years Since Last Promotion**: Enter the number of years since the employee's last promotion.")
st.write("- **Years with Current Manager**: Enter the number of years working under the current manager.")

st.markdown("### 🏢 Other Employee Attributes")
st.write("- **Department**: Select the department where the employee works.")
st.write("- **Business Travel Frequency**: Select how frequently the employee travels for work.")
st.write("- **Education Background**: Choose the educational background of the employee.")
st.write("- **Training Times Last Year**: Enter how many training sessions the employee attended in the last year.")
st.write("- **Gender**: Select the employee’s gender.")
st.write("- **Attrition Status**: Choose whether the employee has left the company or is still employed.")
st.markdown("---")

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
    input_df = pd.DataFrame([input_data])
    feature_order = list(input_df.columns)
    prediction = model.predict(input_df)[0]
    rating_text, rating_description = PERFORMANCE_RATING.get(prediction, (prediction, "Unknown"))
    st.subheader("Prediction Result")
    st.metric("Predicted Performance Rating", f"{rating_text} ({prediction})")
    st.write(f"**What this means:** {rating_description}")
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({"Feature": feature_order, "Importance": model.feature_importances_})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(y=importance_df["Feature"], x=importance_df["Importance"], palette="viridis", ax=ax)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance for Performance Prediction")
        st.pyplot(fig)
