import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib
import os
from googletrans import Translator
from tensorflow.keras.models import load_model

translator = Translator()

languages = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'ur': 'Urdu'
}

selected_language = st.selectbox("Select Language", options=languages.keys(), format_func=lambda x: languages[x])

# Function to translate text
def translate_text(text, lang='en'):
        return text  # Return original text if translation fails


# Load the saved model
model = load_model('disease_prediction_model_2.h5')

# Load the encoders for animal and symptom
animal_encoder = joblib.load('animal_encoder.pkl')  # Save your encoder with joblib
symptom_encoder = joblib.load('symptom_encoder.pkl')  # Save your encoder with joblib
disease_encoder = joblib.load('disease_encoder.pkl')  # Save your encoder with joblib


# Define the prediction function
def predict_disease(animal_name, symptom_name):
    try:
        # Check if animal_name exists in encoder classes
        if animal_name in animal_encoder.classes_:
            animal_encoded = animal_encoder.transform([animal_name]).reshape(1, -1)
        else:
            # Handle unseen animal by using a zero-vector with the correct length
            animal_encoded = np.zeros((1, len(animal_encoder.classes_)))  # Zero-vector for unknown category
            st.warning(f"Unknown animal: {animal_name}. Using default encoding.")
    except Exception as e:
        st.error(f"Error processing animal: {e}")
        animal_encoded = np.zeros((1, len(animal_encoder.classes_)))

    try:
        # Check if symptom_name exists in encoder classes
        if symptom_name in symptom_encoder.classes_:
            symptom_encoded = symptom_encoder.transform([symptom_name]).reshape(1, -1)
        else:
            # Handle unseen symptom by using a zero-vector with the correct length
            symptom_encoded = np.zeros((1, len(symptom_encoder.classes_)))  # Zero-vector for unknown category
            st.warning(f"Unknown symptom: {symptom_name}. Using default encoding.")
    except Exception as e:
        st.error(f"Error processing symptom: {e}")
        symptom_encoded = np.zeros((1, len(symptom_encoder.classes_)))

    # Combine the encoded features (ensure both are 2D arrays and match the expected length)
    input_features = np.concatenate([animal_encoded, symptom_encoded], axis=1)  # (1, n_animal_features + n_symptom_features)

    # Ensure the input has the correct length (496 features in total)
    expected_input_length = len(animal_encoder.classes_) + len(symptom_encoder.classes_)

    if input_features.shape[1] != expected_input_length:
        # Pad or truncate the input_features to match the expected input length (496)
        input_features = np.pad(input_features, ((0, 0), (0, expected_input_length - input_features.shape[1])), mode='constant')

    # Predict the disease probabilities
    prediction_probabilities = model.predict(input_features)[0]

    # Get the top 3-5 diseases with their probabilities
    top_indices = np.argsort(prediction_probabilities)[::-1][:5]
    top_diseases = disease_encoder.inverse_transform(top_indices)
    top_probabilities = prediction_probabilities[top_indices]

    # Prepare the predictions to display
    predictions = []
    for disease, prob in zip(top_diseases, top_probabilities):
        probability_percentage = f"{prob * 100:.2f}%"
        if prob == 0:
            probability_percentage = "<10%"  # If the probability is 0, display "<10%"
        predictions.append(f"{disease}: {probability_percentage}")

    # Get the disease with the highest probability
    max_prob_idx = top_indices[0]
    predicted_disease = disease_encoder.inverse_transform([max_prob_idx])[0]

    return predictions, predicted_disease


# Function to log input and output to an Excel file
def log_data(animal_name, symptom_name, disease):
    # Define the file name
    file_name = 'disease_predictions_log.xlsx'

    # Create a dictionary for the data
    data = {
        'Animal': [animal_name],
        'Symptom': [symptom_name],
        'Disease': [disease]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Check if the file exists
    if os.path.exists(file_name):
        # Append the data to the existing file
        existing_data = pd.read_excel(file_name)
        updated_data = pd.concat([existing_data, df], ignore_index=True)
        updated_data.to_excel(file_name, index=False)
    else:
        # Create a new file and save the data
        df.to_excel(file_name, index=False)


# Streamlit UI
st.title(translate_text('Animal Disease Prediction', selected_language))
st.write(translate_text("Enter the animal name and symptom to predict the disease:", selected_language))

# User inputs
animals = ["cow", "buffalo", "sheep", "goat"]

# Create a selectbox for animal input
any_animal_name = st.selectbox(translate_text("Select an Animal", selected_language), animals)
any_symptom_name = st.text_input(translate_text('Symptom', selected_language), '')

animal_name = translate_text(any_animal_name, 'en')
symptom_name = translate_text(any_symptom_name, 'en')

# Prediction on button click
if st.button(translate_text('Predict', selected_language)):
    if animal_name and symptom_name:
        predictions, predicted_disease = predict_disease(animal_name, symptom_name)

        st.write(translate_text("Top 3-5 possible diseases:", selected_language))
        for prediction in predictions:
            st.write(prediction)

        # Log the data to the Excel file
        log_data(animal_name, symptom_name, predicted_disease)

        with st.expander(translate_text("Know More", selected_language)):
            st.markdown("[Know more about your disease](https://agrovetcare-yqz3vvwra2bveydzyzqlsq.streamlit.app/Education)")

        with st.expander(translate_text("Visit Marketplace", selected_language)):
            st.markdown("[Visit Amazon Marketplace](https://www.amazon.in)")

        with st.expander(translate_text("Contact Experts", selected_language)):
            # Expert 1
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
            with col1:
                st.markdown(f"""
                *Name*: Dr. Singh  
                *Contact*: [9876543211](tel:9876543211)  
                *Status*: :green[Online]  
                """)
                contact_form = f"""
                <form action="https://formsubmit.co/arnabgupta983@gmail.com" method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="animal" value="{animal_name}" placeholder="{translate_text('Animal Name', selected_language)}" required>
                    <input type="text" name="symptom" value="{symptom_name}" placeholder="{translate_text('Symptom', selected_language)}" required>
                    <textarea name="message" placeholder="{translate_text('Tell your problem', selected_language)}"></textarea>
                    <button type="submit">{translate_text('Send', selected_language)}</button>
                </form>
                """
                st.markdown(contact_form, unsafe_allow_html=True)

            with col2:
                st.image("manavatar.png", width=50)  # Adjust the width and image path

            st.markdown("---")  # Horizontal separator

            # Expert 2
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
            with col1:
                st.markdown(f"""
                    *Name*: Dr. Sharma  
                    *Contact*: [1234567899](tel:1234567899)  
                    *Status*: :red[Offline]  
                    """)
                contact_form = f"""
                <form action="https://formsubmit.co/arnabgupta983@gmail.com" method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="animal" value="{animal_name}" placeholder="{translate_text('Animal Name', selected_language)}" required>
                    <input type="text" name="symptom" value="{symptom_name}" placeholder="{translate_text('Symptom', selected_language)}" required>
                    <textarea name="message" placeholder="{translate_text('Tell your problem', selected_language)}"></textarea>
                    <button type="submit">{translate_text('Send', selected_language)}</button>
                </form>
                """
                st.markdown(contact_form, unsafe_allow_html=True)

            with col2:
                st.image("womanavatar.png", width=50)  # Adjust the width and image path

    else:
        st.warning(translate_text("Please enter both animal name and symptom.", selected_language))
