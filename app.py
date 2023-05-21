import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Load the saved model
model = joblib.load('email_class.sav')

# Initialize vectorizer and tfidf transformer
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def main():
    # Streamlit app code goes here


# Load the saved model
model = joblib.load('email_class.sav')

def main():
    # Streamlit app code goes here
    st.title("Custom ML Model Deployment")

    # Input for user
    user_input = st.text_input("Enter your input:")

    # Process user input and make prediction
    if st.button("Predict"):
        # Preprocess the input data
        input_vector = vectorizer.transform([user_input])
        input_tfidf = tfidf_transformer.transform(input_vector)

        # Reshape the input to a 2D array
        input_reshaped = input_tfidf.reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_reshaped)

        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
