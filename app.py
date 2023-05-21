import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the entire dataset from the CSV file
data = pd.read_csv('final_dataset.csv')

# Filter the dataset for the three classes
fraud_data = data[data['Label'] == 'FRAUD'][:1000]
normal_data = data[data['Label'] == 'NORMAL'][:1000]
spam_data = data[data['Label'] == 'SPAM'][:1000]

# Concatenate the data for the three classes
train_data = pd.concat([fraud_data, normal_data, spam_data])

# Split the dataset into training and testing sets
train_emails = train_data['Email']
train_labels = train_data['Label']

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the training data
train_vectors = vectorizer.fit_transform(train_emails)

# Train the model
model = LogisticRegression()
model.fit(train_vectors, train_labels)

def main():
    # Streamlit app code goes here
    st.title("Custom Text Classification")

    # Input for user
    user_input = st.text_input("Enter your input:")

    # Process user input and make prediction
    if st.button("Predict"):
        # Vectorize the user input
        input_vector = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_vector)

        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
