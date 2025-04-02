import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score

# ✅ File uploader to load dataset manually
st.title("📚 AI-Powered Student Grading and Fine-Tuning")
uploaded_file = st.file_uploader("Upload Behavioral Economics Dataset (CSV)", type="csv")

# Load the dataset for fine-tuning
def load_and_process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Prepare the dataset for fine-tuning
    if "Concept" in df.columns and "Student_Response" in df.columns and "Faculty_Grade" in df.columns:
        # Map grades to numeric labels (e.g., 'A+' -> 0, 'A' -> 1, etc.)
        grade_mapping = {"A+": 0, "A": 1, "A-": 2, "B+": 3, "B": 4, "B-": 5, "C+": 6, "C": 7, "C-": 8, "D": 9, "F": 10}
        df['Grade_Label'] = df['Faculty_Grade'].map(grade_mapping)
        
        # Split the data into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Convert to Hugging Face Dataset format
        train_dataset = Dataset.from_pandas(train_df[['Concept', 'Student_Response', 'Grade_Label']])
        test_dataset = Dataset.from_pandas(test_df[['Concept', 'Student_Response', 'Grade_Label']])

        return train_dataset, test_dataset, grade_mapping
    else:
        st.error("🚨 'Concept', 'Student_Response', or 'Faculty_Grade' column not found in dataset! Please check the file.")
        st.stop()

# Load tokenizer and model
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=11)  # 11 grades (0 to 10)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['Concept'] + " " + examples['Student_Response'], padding="max_length", truncation=True)

# Fine-tune the model
def fine_tune_model(train_dataset, test_dataset):
    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {'accuracy': accuracy_score(p.predictions.argmax(-1), p.label_ids)}
    )
    
    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

# Function to predict grade (after fine-tuning)
def predict_grade(concept, student_response):
    combined_input = f"Concept: {concept}. Student Answer: {student_response}"
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class (grade) index
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    grade_mapping = {0: "A+", 1: "A", 2: "A-", 3: "B+", 4: "B", 5: "B-", 6: "C+", 7: "C", 8: "C-", 9: "D", 10: "F"}
    predicted_grade = grade_mapping.get(predicted_class, "Unknown")
    return predicted_grade

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

# Streamlit UI for Concept Selection
st.write("Select the concept from the dropdown and enter the student's response to predict their grade.")

# Concept selection dropdown
concept = st.selectbox("🧠 Select Concept", unique_concepts)
    
# Student response input
student_answer = st.text_area("📝 Student's Answer", height=150)

# Button to fine-tune model
if st.button("🎯 Fine-tune Model with New Dataset"):
    if uploaded_file:
        train_dataset, test_dataset, grade_mapping = load_and_process_data(uploaded_file)
        fine_tune_model(train_dataset, test_dataset)
        st.success("✅ Model fine-tuned successfully!")
else:
    # If the model is already fine-tuned, allow grade prediction and feedback
    if st.button("🎯 Predict Grade and Get Feedback"):
        if student_answer:
            predicted_grade = predict_grade(concept, student_answer)  # Predict grade
            feedback = generate_feedback(predicted_grade)  # Generate feedback

            # Display the predicted grade and feedback
            st.success(f"✅ Predicted Grade: **{predicted_grade}**")
            st.write(f"📋 **Feedback**: {feedback}")
        else:
            st.warning("⚠️ Please enter the student's response.")
