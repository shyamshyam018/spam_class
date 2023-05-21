import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Loading the model
model = joblib.load('email_class.sav')

# Load the dataset from the CSV file
df = pd.read_csv('final_dataset.csv', encoding='latin-1')

# Extract the email texts and labels from the dataset
emails = df['Email'].tolist()
labels = df['Label'].tolist()

# Create and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(emails)

# Save the vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

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
    message = st_text_input(label='Enter the message', key='message_input')

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
