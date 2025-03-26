# CPAI-Project
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer (you need to have the model saved in the "grading_model" folder)
model_path = "./grading_model"  # If using a different path, change it accordingly
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define grade mapping (ensure it matches the training labels)
grade_mapping = {0: "A+", 1: "A", 2: "B", 3: "C", 4: "D", 5: "F"}

# Function to predict grade
def predict_grade(student_response):
    inputs = tokenizer(student_response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return grade_mapping[predicted_class]

# Streamlit UI
st.title("Student Grade Prediction")
st.write("Enter the student's response to predict the grade:")

# Input text box
student_answer = st.text_area("Student's Answer", height=200)

if st.button("Predict Grade"):
    if student_answer:
        predicted_grade = predict_grade(student_answer)
        st.success(f"Predicted Grade: {predicted_grade}")
    else:
        st.warning("Please enter a student response.")
