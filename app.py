import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Header
st.title('📧 Email Spam Classifier')
st.write('Welcome to the Email Spam Classifier! This application helps you determine if an email is spam or not.')

# Instructions
st.subheader('Instructions:')
st.write('1. Enter the email text in the input box below.')
st.write('2. Click the "Classify" button to check if the email is spam.')

# User input
user_input = st.text_area('✉️ Enter your email here', height=150)

# Classify button
if st.button("🔍 Classify"):
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0] == 0:
            st.success("✅ The email is not spam")
        else:
            st.error("🚫 The email is spam")
    else:
        st.warning("⚠️ Please enter an email to classify")

# Footer
st.write('---')
st.write('Developed by Om Shukla')
st.write('For more information, visit our https://github.com/omshukla12/Spam-Classifier')
st.write('Contact us at omshukla2060@gmail.com')