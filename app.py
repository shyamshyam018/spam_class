import streamlit as st
import joblib

# Load the saved model
model = joblib.load('email_class.sav')

def main():
    # Streamlit app code goes here
    st.title("Custom ML Model Deployment")

    # Input for user
    user_input = st.text_input("Enter your input:")

    # Process user input and make prediction
    if st.button("Predict"):
        user_input = user_input.reshape(-1, 1)
        prediction = model.predict([user_input])
        
        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
