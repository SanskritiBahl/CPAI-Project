# CPAI-Project
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from huggingface_hub import login

# Authenticate Hugging Face if you're using a private model
login(token=st.secrets["huggingface"]["token"])

# Model path (local or Hugging Face model name)
# Example Hugging Face model name (replace with your model if needed)
model_path = "bert-base-uncased"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Grade mapping (for your use case)
grade_mapping = {0: "A+", 1: "A", 2: "B", 3: "C", 4: "D", 5: "F"}

# Load and display the CSV file when uploaded
@st.cache_data
def load_data(uploaded_file):
    # Load the dataset CSV into a pandas dataframe
    return pd.read_csv(uploaded_file)

# Function to predict grade
def predict_grade(concept, student_response):
    # Format the input text more explicitly
    combined_input = f"Concept: {concept}. Student's answer: {student_response}"

    # Tokenize the combined input (ensuring padding and truncation)
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    return grade_mapping.get(predicted_class, "Unknown Grade")

# Streamlit UI
st.title("Student Grade Prediction")
st.write("Enter the concept and the student's response to predict the grade:")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

# Display the contents of the uploaded file (optional)
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write(data.head())  # Display the first few rows of the dataset

    # Concept input (let's assume dataset has a column 'Concept')
    concept = st.selectbox("Select Concept", options=data["Concept"].unique())

    # Input box for student’s response
    student_answer = st.text_area("Student's Answer", height=200)

    # Button to trigger prediction
    if st.button("Predict Grade"):
        if student_answer:
            predicted_grade = predict_grade(concept, student_answer)
            st.success(f"Predicted Grade: {predicted_grade}")
        else:
            st.warning("Please enter a student response.")
else:
    st.warning("Please upload a CSV file with the dataset.")


