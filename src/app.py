import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered"
)

@st.cache_resource
def load_model(path="best_model.pkl"):
    return joblib.load(path)

model = load_model()

st.title("💼 Predict Employee Salary Bracket")
st.write(
    """
    Enter employee details below to predict whether their salary exceeds \$50K annually.
    """
)

with st.form("employee_input_form"):
    st.header("🧾 Employee Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 75, 30)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=0, value=100000)
        educational_num = st.slider("Education Num (5–16)", 5, 16, 10)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
        hours_per_week = st.slider("Hours worked per week", 35, 50, 40)

    with col2:
        workclass = st.selectbox(
            "Workclass",
            ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov", "State-gov", "NotListed"]
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"]
        )
        occupation = st.selectbox(
            "Occupation",
            [
                "Tech-support", "Craft-repair", "Other-service", "Sales",
                "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                "Transport-moving", "Priv-house-serv", "Protective-serv",
                "Others"
            ]
        )
        relationship = st.selectbox(
            "Relationship",
            ["Husband", "Not-in-family", "Own-child", "Unmarried", "Other-relative"]
        )
        race = st.selectbox(
            "Race",
            ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
        )
        gender = st.radio("Gender", ["Male", "Female"])
        native_country = st.selectbox(
            "Native Country",
            ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Others"]
        )

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    features = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "fnlwgt": [fnlwgt],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "native-country": [native_country],
        "educational-num": [educational_num],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week]
    })

    st.write("### ✅ Input Preview")
    st.dataframe(features)

    prediction = model.predict(features)
    st.success(f"**Result:** The model predicts this employee earns **{prediction[0]}**.")

st.divider()
st.header("📂 Bulk Prediction")

csv_file = st.file_uploader(
    "Upload a CSV with the same columns as above (EXACT!).",
    type="csv"
)

if csv_file:
    try:
        batch_df = pd.read_csv(csv_file)
        st.write("Sample of uploaded data:", batch_df.head())

        batch_predictions = model.predict(batch_df)
        batch_df["SalaryPrediction"] = batch_predictions

        st.write("### 📊 Predictions")
        st.dataframe(batch_df.head())

        output = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results",
            data=output,
            file_name="salary_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
