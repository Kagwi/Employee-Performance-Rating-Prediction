import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model with error handling
try:
    model = joblib.load('best_model_random_forest.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'best_model_random_forest.pkl' exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Configure page
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("Employee Attrition Prediction")

# Predefined mappings for categorical variables
category_mappings = {
    "Gender": {"Male": 0, "Female": 1},
    "EducationBackground": {"Life Sciences": 0, "Medical": 1, "Technical Degree": 2, "Human Resources": 3, "Other": 4},
    "MaritalStatus": {"Single": 0, "Married": 1, "Divorced": 2},
    "EmpDepartment": {"Sales": 0, "R&D": 1, "HR": 2, "Marketing": 3, "Technical": 4},
    "EmpJobRole": {"Manager": 0, "Researcher": 1, "Sales Executive": 2, "Technician": 3, "HR Specialist": 4},
    "BusinessTravelFrequency": {"Travel_Rarely": 0, "Travel_Frequently": 1, "Non-Travel": 2},
    "OverTime": {"Yes": 1, "No": 0}
}

# Create input sections
with st.form("employee_details"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        EmpNumber = st.number_input("Employee Number", min_value=0)
        Age = st.number_input("Age", min_value=18, max_value=65)
        Gender = st.selectbox("Gender", list(category_mappings["Gender"].keys()))
        EducationBackground = st.selectbox("Education Background", list(category_mappings["EducationBackground"].keys()))
        MaritalStatus = st.selectbox("Marital Status", list(category_mappings["MaritalStatus"].keys()))
        EmpDepartment = st.selectbox("Department", list(category_mappings["EmpDepartment"].keys()))
    
    with col2:
        EmpJobRole = st.selectbox("Job Role", list(category_mappings["EmpJobRole"].keys()))
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", list(category_mappings["BusinessTravelFrequency"].keys()))
        DistanceFromHome = st.number_input("Distance from Home (miles)", min_value=1)
        EmpEducationLevel = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        EmpEnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        EmpHourlyRate = st.number_input("Hourly Rate", min_value=20, max_value=100)
    
    with col3:
        EmpJobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        EmpJobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        EmpJobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        NumCompaniesWorked = st.number_input("Companies Worked At", min_value=0)
        OverTime = st.selectbox("Overtime", list(category_mappings["OverTime"].keys()))
        EmpLastSalaryHikePercent = st.number_input("Salary Hike (%)", min_value=0, max_value=25)
    
    submit_button = st.form_submit_button("Predict Attrition")

# Convert input data into a DataFrame
if submit_button:
    input_data = {
        'Age': Age,
        'Gender': category_mappings['Gender'][Gender],
        'EducationBackground': category_mappings['EducationBackground'][EducationBackground],
        'MaritalStatus': category_mappings['MaritalStatus'][MaritalStatus],
        'EmpDepartment': category_mappings['EmpDepartment'][EmpDepartment],
        'EmpJobRole': category_mappings['EmpJobRole'][EmpJobRole],
        'BusinessTravelFrequency': category_mappings['BusinessTravelFrequency'][BusinessTravelFrequency],
        'DistanceFromHome': DistanceFromHome,
        'EmpEducationLevel': EmpEducationLevel,
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpHourlyRate': EmpHourlyRate,
        'EmpJobInvolvement': EmpJobInvolvement,
        'EmpJobLevel': EmpJobLevel,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': category_mappings['OverTime'][OverTime],
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Ensure feature alignment
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_df.columns
    input_df = input_df.reindex(columns=model_features, fill_value=0)
    
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
        
        st.subheader("Prediction Results")
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric("Predicted Attrition", 
                      value="High Risk" if prediction[0] == 1 else "Low Risk",
                      delta=f"{(probability * 100):.1f}% confidence" if probability else "Confidence unavailable")
        
        with result_col2:
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': model_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance_df.head(10))
            else:
                st.info("Feature importance not available for this model")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.sidebar.markdown("""
**User Guide:**
1. Fill in all employee details
2. Click 'Predict Attrition'
3. View prediction results
4. Check feature importance (if available)
""")
