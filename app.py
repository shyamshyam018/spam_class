import pickle
import streamlit as st

# Load the trained model from a saved file
model = pickle.load(open('spam_classifier.sav', 'rb'))

# Create a Streamlit app that accepts user input and displays the predicted spam classification
def main():
    # Set the app title
    st.title('Spam Classification App')
    
    # Get user input
    input_text = st.text_area('Enter your email content here:')
    
    # Make a prediction using the loaded model
    if input_text:
        prediction = model.predict([input_text])
        if prediction == 1:
            st.warning('This email is classified as SPAM.')
        else:
            st.success('This email is NOT classified as spam.')

if __name__ == '__main__':
    main()
