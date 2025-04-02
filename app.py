import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ File uploader to load dataset manually
st.title("📚 AI-Powered Student Grading")
uploaded_file = st.file_uploader("Upload Behavioral Economics Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Show dataset after upload
    st.write("Here is the dataset you've uploaded:")
    st.dataframe(df)  # Displays the uploaded dataset as a table

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

    # ✅ Function to generate feedback based on grade
    def generate_feedback(grade):
        feedback_mapping = {
            "A+": "Excellent work! You have a deep understanding of the concept.",
            "A": "Great job! You're very close to mastering this concept.",
            "A-": "Good work, but there's room for improvement. Review the key concepts.",
            "B+": "You're on the right track, but you might need more practice with the concept.",
            "B": "Solid effort, but you need to focus on understanding the underlying principles.",
            "B-": "You need to work on understanding the core concept more clearly.",
            "C+": "Your response shows some understanding, but there are key areas that need attention.",
            "C": "You have basic knowledge of the topic, but additional study is needed.",
            "C-": "You're struggling with the core concepts. Please seek additional help.",
            "D": "Your answer needs improvement. Consider revisiting the material.",
            "F": "Unfortunately, your understanding of the concept is insufficient. Please review the content again."
        }
        return feedback_mapping.get(grade, "No feedback available")

    # ✅ Streamlit UI for Concept Selection
    st.write("Select the concept from the dropdown and enter the student's response to predict their grade.")

    concept = st.selectbox("🧠 Select Concept", unique_concepts)
    student_answer = st.text_area("📝 Student's Answer", height=150)

    if st.button("🎯 Predict Grade"):
        if student_answer:
            predicted_grade = predict_grade(concept, student_answer)
            feedback = generate_feedback(predicted_grade)
            st.success(f"✅ Predicted Grade: **{predicted_grade}**")
            st.write(f"📋 **Feedback**: {feedback}")
        else:
            st.warning("⚠️ Please enter the student's response.")
else:
    st.warning("⚠️ Please upload the dataset to proceed.")
