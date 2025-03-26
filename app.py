# CPAI-Project
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import login

# If you're using a Hugging Face model or private models, authenticate with the token (Streamlit Secrets)
login(token=st.secrets["huggingface"]["token"])

# Model path (local or Hugging Face model name)
# If you are using a Hugging Face model from the Hub, just use the model name like 'bert-base-uncased'
model_path = "bert-base-uncased"  # Example Hugging Face model name

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Grade mapping (for your use case)
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

# Input box for student’s response
student_answer = st.text_area("Student's Answer", height=200)

# Button to trigger prediction
if st.button("Predict Grade"):
    if student_answer:
        predicted_grade = predict_grade(student_answer)
        st.success(f"Predicted Grade: {predicted_grade}")
    else:
        st.warning("Please enter a student response.")

