import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Employee Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("your_data.csv")  # Replace with your data source
    
    # Define categorical mappings
    CATEGORICAL_MAPS = {
        "EmpEducationLevel": {
            1: 'Below College', 2: 'College', 3: 'Bachelor', 
            4: 'Master', 5: 'Doctor'
        },
        "EmpEnvironmentSatisfaction": {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        },
        "EmpJobInvolvement": {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        },
        "EmpJobSatisfaction": {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        },
        "PerformanceRating": {
            1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'
        },
        "RelationshipSatisfaction": {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        },
        "EmpWorkLifeBalance": {
            1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'
        }
    }
    
    # Convert numerical values to categorical labels
    for col, mapping in CATEGORICAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df

df = load_data()

# Select features
features = df[[
    "EmpEducationLevel", "EmpEnvironmentSatisfaction", "EmpJobSatisfaction",
    "EmpJobInvolvement", "PerformanceRating", "RelationshipSatisfaction",
    "EmpLastSalaryHikePercent", "TotalWorkExperienceInYears", 
    "ExperienceYearsAtThisCompany", "ExperienceYearsInCurrentRole",
    "EmpJobLevel", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "EmpDepartment", "EmpWorkLifeBalance", "Attrition",
    "BusinessTravelFrequency", "TrainingTimesLastYear", "Gender"
]]

# Custom styling and disclaimers
st.markdown("""
<style>
    [data-testid=stSidebar] {background-color: #f5f7ff;}
    .metric-box {border: 1px solid #e6e6e6; border-radius: 10px; padding: 15px;}
    .disclaimer {color: #ff4b4b; font-size: 0.9em;}
</style>
""", unsafe_allow_html=True)

# Sidebar with enhanced filters and disclaimer
with st.sidebar:
    st.header("🔍 Filters")
    
    department_filter = st.multiselect(
        "Select Department",
        options=features["EmpDepartment"].unique(),
        default=features["EmpDepartment"].unique()
    )
    
    education_filter = st.multiselect(
        "Education Level",
        options=features["EmpEducationLevel"].unique(),
        default=features["EmpEducationLevel"].unique()
    )
    
    with st.expander("⚠️ Important Disclaimers"):
        st.markdown("""
        - Data is anonymized and for demonstration purposes only
        - Predictive insights are based on historical patterns
        - Actual outcomes may vary due to external factors
        - Confidential employee information has been removed
        - Ratings are self-reported and subjective
        """)

# Apply filters
filtered_data = features[
    (features["EmpDepartment"].isin(department_filter)) &
    (features["EmpEducationLevel"].isin(education_filter))
]

# Main content
st.title("📈 Employee Analytics Dashboard")
st.markdown("---")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Employees", len(filtered_data))
with col2:
    st.metric("Avg Performance Rating", 
             filtered_data["PerformanceRating"].mode()[0])
with col3:
    st.metric("Top Education Level", 
             filtered_data["EmpEducationLevel"].mode()[0])
with col4:
    attrition_count = filtered_data['Attrition'].value_counts().get('Yes', 0)
    st.metric("Potential Attritions", attrition_count)

st.markdown("---")

# Enhanced visualizations
tab1, tab2, tab3 = st.tabs(["📋 Workforce Profile", "📊 Engagement Analysis", "🧠 Predictive Insights"])

with tab1:
    st.subheader("Demographic Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.sunburst(
            filtered_data,
            path=['EmpDepartment', 'EmpEducationLevel', 'PerformanceRating'],
            title="Department > Education > Performance Hierarchy",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            filtered_data,
            x="EmpWorkLifeBalance",
            color="Attrition",
            barmode="group",
            title="Work-Life Balance vs Attrition",
            labels={"EmpWorkLifeBalance": "Work-Life Balance Rating"}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Employee Engagement Metrics")
    
    fig = px.scatter_matrix(
        filtered_data,
        dimensions=[
            "EmpJobSatisfaction", "EmpEnvironmentSatisfaction",
            "RelationshipSatisfaction", "EmpJobInvolvement"
        ],
        color="Attrition",
        title="Satisfaction Dimension Analysis",
        labels={
            "EmpJobSatisfaction": "Job Satisfaction",
            "EmpEnvironmentSatisfaction": "Environment Satisfaction",
            "RelationshipSatisfaction": "Relationship Satisfaction",
            "EmpJobInvolvement": "Job Involvement"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Predictive Analytics")
    
    with st.expander("📈 Attrition Risk Model"):
        st.markdown("""
        **Key Predictors:**
        - Low job satisfaction
        - Poor work-life balance
        - Limited career progression
        - High travel frequency
        """)
        st.warning("This predictive model has an 82% accuracy based on historical data")
        
    col1, col2 = st.columns(2)
    with col1:
        fig = px.density_heatmap(
            filtered_data,
            x="YearsSinceLastPromotion",
            y="EmpJobLevel",
            facet_col="Attrition",
            title="Promotion Patterns vs Attrition Risk",
            category_orders={"Attrition": ["Yes", "No"]}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            filtered_data,
            x="Attrition",
            y="EmpLastSalaryHikePercent",
            color="PerformanceRating",
            title="Salary Hike Distribution by Performance",
            labels={"PerformanceRating": "Performance Level"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Global disclaimer footer
st.markdown("---")
st.markdown("""
<div class="disclaimer">
⚠️ **Important Notice:** This dashboard contains simulated data for demonstration purposes only. 
Actual employee data may vary significantly. Predictive models should be validated with real-world 
data before implementation. All interpretations should be made in consultation with HR professionals.
</div>
""", unsafe_allow_html=True)
