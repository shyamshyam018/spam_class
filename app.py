import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib

# Training and saving the model
model = ...  # Your trained model
joblib.dump(model, 'email_class.joblib')

# Loading the model
model = joblib.load('email_class.joblib')




# sidebar for navigation
with st.sidebar:
    selected_category = option_menu('Email Classification System',
                                    ['Spam', 'Fraud', 'Normal'],
                                    icons=['exclamation', 'ban', 'check'],
                                    default_index=0)

# Category Selection Page
if selected_category == 'Spam':
    # page title
    st.title('Spam Classification using ML')

    # getting the input data from the user
    user_input = st.text_input('Enter an email')

    # code for prediction
    spam_diagnosis = ''

    # creating a button for prediction
    if st.button('Classify Email'):
        spam_prediction = model.predict([user_input])

        if spam_prediction[0] == 'SPAM':
            spam_diagnosis = 'The email is classified as SPAM'
        else:
            spam_diagnosis = 'The email is not classified as SPAM'

    st.success(spam_diagnosis)


# Category Selection Page
if selected_category == 'Fraud':
    # page title
    st.title('Fraud Classification using ML')

    # getting the input data from the user
    user_input = st.text_input('Enter an email')

    # code for prediction
    fraud_diagnosis = ''

    # creating a button for prediction
    if st.button('Classify Email'):
        fraud_prediction = model.predict([user_input])

        if fraud_prediction[0] == 'FRAUD':
            fraud_diagnosis = 'The email is classified as FRAUD'
        else:
            fraud_diagnosis = 'The email is not classified as FRAUD'

    st.success(fraud_diagnosis)


# Category Selection Page
if selected_category == 'Normal':
    # page title
    st.title('Normal Classification using ML')

    # getting the input data from the user
    user_input = st.text_input('Enter an email')

    # code for prediction
    normal_diagnosis = ''

    # creating a button for prediction
    if st.button('Classify Email'):
        normal_prediction = model.predict([user_input])

        if normal_prediction[0] == 'NORMAL':
            normal_diagnosis = 'The email is classified as NORMAL'
        else:
            normal_diagnosis = 'The email is not classified as NORMAL'

    st.success(normal_diagnosis)
