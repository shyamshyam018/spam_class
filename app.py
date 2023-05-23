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

# Description Page
st.title('Email Classification Model')
st.write('This model classifies emails into different categories based on their content and characteristics.')

# Get Started button
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    if st.button('Get Started'):
        # Email Classification Page
        st.title('Email Classification')
        
        # Input text box
        user_input = st.text_area('Enter the email text', height=200)
        
        # Classify button
        if st.button('Classify'):
            # Transform user input using the vectorizer
            input_vector = vectorizer.transform([user_input])
            
            # Make predictions
            email_class = model.predict(input_vector)[0]
            
            # Display the predicted class with color
            if email_class in ['SPAM', 'FRAUD']:
                st.error('The email is classified as: spam')
            else:
                st.success('The email is classified as: {}'.format(email_class))

# About Us button
if st.button('About Us'):
    # Page title
    st.title('About Us')
    
    # Description
    st.write("Meet the team:")
    st.write("👩‍💼 Chandrika - Btech IT")
    st.write("👩‍💼 Akila - Btech IT")
    st.write("👩‍💼 Swathi - Btech IT")
