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
        ('Email Classification', 'Description', 'About Us')
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

        # Set color based on classification result
        if email_class == 'normal' or email_class == 'important':
            color = 'green'
        else:
            color = 'red'

        # Display the predicted class with color
        st.markdown(f'<font color={color}>The email is classified as: {email_class}</font>', unsafe_allow_html=True)

# Description Page
elif selected == 'Description':
    # Page title
    st.title('Description')

    # Methodology
    st.write('This project uses a Multinomial Naive Bayes classifier for email classification. The text of the email is transformed using a TF-IDF vectorizer and then fed into the classifier to predict the class of the email.')

# About Us Page
elif selected == 'About Us':
    # Page title
    st.title('About Us')

    # Description
    st.write("Meet the team:")
    st.write("👩‍💼 Chandrika - Btech IT")
    st.write("👩‍💼 Akila - Btech IT")
    st.write("👩‍💼 Swathi - Btech IT")
