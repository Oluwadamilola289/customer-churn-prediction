import streamlit as st
import pandas as pd
import joblib
comparison = pd.read_csv("reports/model_comparison.csv")
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)
st.sidebar.title("📊 Customer Churn Prediction")

st.sidebar.success("Production ML Demo")

st.sidebar.metric(
    "Best Accuracy",
    "81.6%"
)

st.sidebar.metric(
    "Customers",
    "7,043"
)

st.sidebar.metric(
    "Features",
    "20"
)

st.sidebar.metric(
    "Algorithms Tested",
    "4"
)

st.sidebar.markdown("---")

st.sidebar.write("Developed by")

st.sidebar.write("**Oluwadamilola Osho**")
st.markdown("---")
st.subheader("📝 Customer Information")
st.write("Enter customer details below to estimate churn probability.")
st.markdown(
"""
### Connect

🔗 GitHub: https://github.com/Oluwadamilola289

💼 LinkedIn: https://www.linkedin.com/in/damilola-e-osho
"""
)
pipeline = joblib.load("models/churn_prediction_pipeline.pkl")

st.title("📞 Telecom Customer Churn Prediction")

st.write(
"""
This interactive application predicts whether a telecom customer is likely to churn using a
 production-ready machine learning pipeline. The model was trained on over 7,000 customer
 records and compares multiple classification algorithms before selecting the best-performing model.
"""
)

st.write(
    "Predict whether a telecom customer is likely to churn."
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0,1]
)

partner = st.selectbox(
    "Partner",
    ["Yes","No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone = st.selectbox(
    "Phone Service",
    ["Yes","No"]
)

multiple = st.selectbox(
    "Multiple Lines",
    ["Yes","No","No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    value=70.0
)

total = st.number_input(
    "Total Charges",
    value=500.0
)

if st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True
    ):

    customer = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone],
        "MultipleLines":[multiple],
        "InternetService":[internet],
        "OnlineSecurity":[security],
        "OnlineBackup":[backup],
        "DeviceProtection":[device],
        "TechSupport":[support],
        "StreamingTV":[tv],
        "StreamingMovies":[movies],
        "Contract":[contract],
        "PaperlessBilling":[paperless],
        "PaymentMethod":[payment],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total]

    })

    prediction = pipeline.predict(customer)[0]

    probability = pipeline.predict_proba(customer)[0][1]

    if prediction == 1:

        st.error("⚠ High Risk Customer")

        st.warning(
       "Retention campaign recommended."
       )

    else:

        st.success("✅ Customer Likely To Stay")


    st.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )
    st.progress(probability)
    confidence = max(probability, 1-probability)

    st.metric(
    "Prediction Confidence",
    f"{confidence:.1%}"
)

with st.expander("📖 About this Project"):

    st.write("""
    This project predicts telecom customer churn using supervised machine learning.

    Pipeline:

    • Data Cleaning
    • Feature Engineering
    • Model Training
    • Model Comparison
    • Prediction Dashboard

    Dataset Size:
    7,043 customers

    Best Model:
    Logistic Regression

    Accuracy:
    81.6%
    """)

st.markdown("---")

st.caption("Built with Python • Scikit-Learn • Pandas • Streamlit")

st.markdown("---")

st.subheader("📊 Model Comparison")

chart = comparison.set_index("Model")

st.bar_chart(chart)

st.dataframe(
    comparison,
    use_container_width=True
)