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
