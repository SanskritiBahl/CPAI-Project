import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ File uploader to load dataset manually
st.title("📚 AI-Powered Student Grading")
uploaded_file = st.file_uploader("behavioral_economics_dataset.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Ensure 'Concept' column exists
    if "Concept" in df.columns:
        unique_concepts = df["Concept"].dropna().unique().tolist()
    else:
        st.error("🚨 'Concept' column not found in dataset! Please check the file.")
        st.stop()

    # ✅ Load model & tokenizer (with error handling)
    MODEL_NAME = "bert-base-uncased"

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)
    except Exception as e:
        st.error(f"🚨 Model loading failed! Error: {e}")
        st.stop()

    # ✅ Grade mapping
    grade_mapping = {0: "A+", 1: "A", 2: "A-", 3: "B+", 4: "B", 5: "B-", 6: "C+", 7: "C", 8: "D", 9: "F"}

    # ✅ Function to predict grade
    def predict_grade(concept, student_response):
        combined_input = f"Concept: {concept}. Student Answer: {student_response}"
        inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return grade_mapping.get(predicted_class, "Unknown")

    # ✅ Streamlit UI for Concept Selection
    st.write("Select the concept from the dropdown and enter the student's response to predict their grade.")

    concept = st.selectbox("🧠 Select Concept", unique_concepts)
    student_answer = st.text_area("📝 Student's Answer", height=150)

    if st.button("🎯 Predict Grade"):
        if student_answer:
            predicted_grade = predict_grade(concept, student_answer)
            st.success(f"✅ Predicted Grade: **{predicted_grade}**")
        else:
            st.warning("⚠️ Please enter the student's response.")
else:
    st.warning("⚠️ Please upload the dataset to proceed.")
