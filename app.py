import streamlit as st
import pandas as pd

# ✅ File uploader to load dataset manually
st.title("📚 AI-Powered Student Grading and Feedback")
uploaded_file = st.file_uploader("Upload Behavioral Economics Dataset (CSV)", type="csv")

# Grade mapping for feedback
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

# Streamlit UI for Concept Selection and Feedback Display
if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Show the dataset after it is uploaded
    st.write("Here is the dataset you've uploaded:")
    st.dataframe(df)  # Displays the uploaded dataset as a table

    # ✅ Ensure 'Concept' column exists
    if "Concept" in df.columns and "Student_Response" in df.columns and "Faculty_Grade" in df.columns:
        unique_concepts = df["Concept"].dropna().unique().tolist()
    else:
        st.error("🚨 Required columns ('Concept', 'Student_Response', 'Faculty_Grade') not found in dataset! Please check the file.")
        st.stop()

    # Concept selection dropdown
    concept = st.selectbox("🧠 Select Concept", unique_concepts)
    
    # Student response input
    student_answer = st.text_area("📝 Student's Answer", height=150)

    # Button to predict grade and feedback
    if st.button("🎯 Predict Grade and Get Feedback"):
        if student_answer:
            # Match student response with the dataset to get the exact grade
            matched_row = df[(df["Concept"] == concept) & (df["Student_Response"].str.lower() == student_answer.lower())]

            if not matched_row.empty:
                # Extract the predicted grade from the matched row
                predicted_grade = matched_row["Faculty_Grade"].values[0]
                feedback = feedback_mapping.get(predicted_grade, "No feedback available")

                # Display the predicted grade and feedback
                st.success(f"✅ Predicted Grade: **{predicted_grade}**")
                st.write(f"📋 **Feedback**: {feedback}")
            else:
                st.warning("⚠️ No matching response found in the dataset.")
        else:
            st.warning("⚠️ Please enter the student's response.")
else:
    st.warning("⚠️ Please upload the dataset to proceed.")
