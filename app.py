import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model_random_forest.pkl')

# Configure page
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("Employee Attrition Prediction")

# Create input sections
with st.form("employee_details"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        EmpNumber = st.number_input("Employee Number", min_value=0)
        Age = st.number_input("Age", min_value=18, max_value=65)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        EducationBackground = st.selectbox("Education Background", 
            ["Life Sciences", "Medical", "Technical Degree", "Human Resources", "Other"])
        MaritalStatus = st.selectbox("Marital Status", 
            ["Single", "Married", "Divorced"])
        EmpDepartment = st.selectbox("Department", 
            ["Sales", "R&D", "HR", "Marketing", "Technical"])
        
    with col2:
        EmpJobRole = st.selectbox("Job Role", 
            ["Manager", "Researcher", "Sales Executive", "Technician", "HR Specialist"])
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", 
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        DistanceFromHome = st.number_input("Distance from Home (miles)", min_value=1)
        EmpEducationLevel = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        EmpEnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        EmpHourlyRate = st.number_input("Hourly Rate", min_value=20, max_value=100)
        
    with col3:
        EmpJobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        EmpJobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        EmpJobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        NumCompaniesWorked = st.number_input("Companies Worked At", min_value=0)
        OverTime = st.selectbox("Overtime", ["Yes", "No"])
        EmpLastSalaryHikePercent = st.number_input("Salary Hike (%)", min_value=0, max_value=25)

    # Additional inputs
    st.subheader("Additional Information")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        EmpRelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        TotalWorkExperienceInYears = st.number_input("Total Work Experience (Years)", min_value=0)
        
    with col5:
        TrainingTimesLastYear = st.number_input("Trainings Last Year", min_value=0)
        EmpWorkLifeBalance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        
    with col6:
        ExperienceYearsAtThisCompany = st.number_input("Years at Company", min_value=0)
        ExperienceYearsInCurrentRole = st.number_input("Years in Current Role", min_value=0)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0)
        YearsWithCurrManager = st.number_input("Years with Current Manager", min_value=0)
        PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])

    submit_button = st.form_submit_button("Predict Attrition")

# Create feature dictionary
if submit_button:
    input_data = {
        'EmpNumber': EmpNumber,
        'Age': Age,
        'Gender': Gender,
        'EducationBackground': EducationBackground,
        'MaritalStatus': MaritalStatus,
        'EmpDepartment': EmpDepartment,
        'EmpJobRole': EmpJobRole,
        'BusinessTravelFrequency': BusinessTravelFrequency,
        'DistanceFromHome': DistanceFromHome,
        'EmpEducationLevel': EmpEducationLevel,
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpHourlyRate': EmpHourlyRate,
        'EmpJobInvolvement': EmpJobInvolvement,
        'EmpJobLevel': EmpJobLevel,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': OverTime,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'EmpRelationshipSatisfaction': EmpRelationshipSatisfaction,
        'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'PerformanceRating': PerformanceRating
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    # Display results
    st.subheader("Prediction Results")
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.metric("Predicted Attrition", 
                 value="High Risk" if prediction[0] == 1 else "Low Risk",
                 delta=f"{probability:.1%} confidence")
    
    with result_col2:
        st.write("**Feature Importance**")
        # Add feature importance visualization if available
        # Note: Requires model to have feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df.head(10))
        else:
            st.info("Feature importance not available for this model")

# Add documentation
st.sidebar.markdown("""
**User Guide:**
1. Fill in all employee details
2. Click 'Predict Attrition'
3. View prediction results
4. Check feature importance (if available)
""")
