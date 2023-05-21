import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the model
model = joblib.load('email_class.sav')

# Loading the vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Select an option',
        ('Email Classification', 'Spam Detection')
    )

# Email Classification Page
if selected == 'Email Classification':
    # Page title
    st.title('Email Classification')

    # Input text box
    user_input = st.text_area('Enter the email text', height=200)

    # Classify button
    if st.button('Classify'):
        # Transform user input using the vectorizer
        input_vector = vectorizer.transform([user_input])

        # Make predictions
        email_class = model.predict(input_vector)[0]

        # Display the predicted class
        st.success(f'The email is classified as: {email_class}')

# Spam Detection Page
elif selected == 'Spam Detection':
    # Page title
    st.title('Spam Detection')

    # Input text box
    message = st.text_area('Enter the message')

    # Predict button
    if st.button('Predict'):
        # Transform user input using the vectorizer
        input_vector = vectorizer.transform([message])

        # Make predictions
        spam_prediction = model.predict(input_vector)[0]

        # Display the prediction result
        if spam_prediction == 1:
            st.warning('This is a spam message.')
        else:
            st.success('This is not a spam message.')
