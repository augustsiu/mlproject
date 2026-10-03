import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Student Exam Performance Indicator")

st.title("Student Exam Performance Indicator")
st.subheader("Student Exam Performance Prediction")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["male", "female"], index=None,
                          placeholder="Select your Gender")
    ethnicity = st.selectbox("Race or Ethnicity",
                             ["group A", "group B", "group C", "group D", "group E"],
                             index=None, placeholder="Select Ethnicity")
    parental_level_of_education = st.selectbox(
        "Parental Level of Education",
        ["associate's degree", "bachelor's degree", "high school",
         "master's degree", "some college", "some high school"],
        index=None, placeholder="Select Parent Education")
    lunch = st.selectbox("Lunch Type", ["free/reduced", "standard"], index=None,
                         placeholder="Select Lunch Type")
    test_preparation_course = st.selectbox("Test preparation Course", ["none", "completed"],
                                           index=None, placeholder="Select Test_course")
    reading_score = st.number_input("Reading Score out of 100", min_value=0, max_value=100,
                                    value=None, placeholder="Enter your Reading Score")
    writing_score = st.number_input("Writing Score out of 100", min_value=0, max_value=100,
                                    value=None, placeholder="Enter your Writing Score")

    submitted = st.form_submit_button("Predict your Math Score")

if submitted:
    fields = [gender, ethnicity, parental_level_of_education, lunch,
              test_preparation_course, reading_score, writing_score]
    if any(f is None for f in fields):
        st.warning("Please fill in all fields.")
    else:
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        st.success(f"The prediction is {results[0]}")
