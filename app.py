import os
import streamlit as st
import pandas as pd
import joblib

from model_utils import (
    DATA_FILE,
    MODEL_FILE,
    load_student_data,
    train_student_model
)

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def get_model_and_data():
    # Use saved files locally when available.
    if os.path.exists(MODEL_FILE) and os.path.exists(DATA_FILE):
        model = joblib.load(MODEL_FILE)
        data = pd.read_csv(DATA_FILE)
        return model, data

    # On Streamlit Cloud, train automatically if the generated files
    # have not been uploaded to the repository.
    data = load_student_data()
    model, accuracy, report, X, data = train_student_model(data)
    data.to_csv(DATA_FILE, index=False)
    return model, data


st.title("🎓 Student Performance Prediction System")
st.caption(
    "A classroom machine-learning prototype that explores possible "
    "academic support needs using historical student data."
)

try:
    with st.spinner("Loading the dataset and machine-learning model..."):
        model, data = get_model_and_data()
except Exception as error:
    st.error("The application could not load the dataset or train the model.")
    st.write("Please check your internet connection and installed requirements.")
    st.exception(error)
    st.stop()

# Dataset overview
st.header("Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total student records", f"{len(data):,}")
col2.metric("Average final grade", f"{data['G3'].mean():.2f}")
col3.metric(
    "Historical at-risk records",
    f"{int((data['G3'] < 10).sum()):,}"
)

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.info(
    "Prediction rule: G3 below 10 means AT RISK. G3 is the final-grade "
    "target and is not included among the prediction input columns."
)

# Sample input file
st.header("Predict student records")
st.write(
    "Upload a CSV file containing the same input columns used by the model. "
    "Do not include the G3 final-grade column."
)

feature_columns = list(model.feature_names_in_)
sample_input = data.drop(columns=["G3"], errors="ignore")[feature_columns].head(5)

st.download_button(
    label="Download sample input CSV",
    data=sample_input.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Choose a student CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)

        missing_columns = [
            column for column in feature_columns
            if column not in new_data.columns
        ]

        if missing_columns:
            st.error(
                "Your CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )
        elif new_data.empty:
            st.warning("The uploaded CSV has no student records.")
        else:
            # Extra columns are ignored; only the trained input columns are used.
            prediction_input = new_data[feature_columns]
            predictions = model.predict(prediction_input)

            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if value == 1 else "NOT AT RISK"
                for value in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)

            at_risk_count = int((predictions == 1).sum())
            not_at_risk_count = int((predictions == 0).sum())

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Predicted AT RISK", at_risk_count)
            result_col2.metric("Predicted NOT AT RISK", not_at_risk_count)

            st.download_button(
                label="Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error("The uploaded file could not be processed. Please check its format.")
        st.exception(error)

st.divider()
st.subheader("Important limitation")
st.write(
    "This is a classroom demonstration using historical data from a "
    "Portuguese secondary school. Its predictions may not be accurate for "
    "other schools or student populations. A prediction is not a diagnosis "
    "and must never be the only basis for decisions about a student. "
    "Faculty should review each student's situation and provide appropriate support."
)
