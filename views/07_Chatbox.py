import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

# Load the saved model
model = tf.keras.models.load_model('disease_prediction_model.h5')

# Load the encoders for animal and symptom
animal_encoder = joblib.load('animal_encoder.pkl')  # Save your encoder with joblib
symptom_encoder = joblib.load('symptom_encoder.pkl')  # Save your encoder with joblib
disease_encoder = joblib.load('disease_encoder.pkl')  # Save your encoder with joblib


# Define the prediction function
def predict_disease(animal_name, symptom_name):
    try:
        # Convert animal name to the correct format (reshape to 2D)
        animal_encoded = animal_encoder.transform([animal_name]).reshape(1, -1)
    except ValueError:
        # Handle unseen animal by using a vector of zeros with the correct length
        animal_encoded = np.zeros((1, len(animal_encoder.classes_)))  # Zero-vector for unknown category
        #st.warning(f"Warning: '{animal_name}' is not in the training data. Using 'unknown' category.")
    
    try:
        # Convert symptom to the correct format (reshape to 2D)
        symptom_encoded = symptom_encoder.transform([symptom_name]).reshape(1, -1)
    except ValueError:
        # Handle unseen symptom by using a vector of zeros with the correct length
        symptom_encoded = np.zeros((1, len(symptom_encoder.classes_)))  # Zero-vector for unknown category
        #st.warning(f"Warning: '{symptom_name}' is not in the training data. Using 'unknown' category.")
    
    # Combine the encoded features (ensure both are 2D arrays and match the expected length)
    input_features = np.concatenate([animal_encoded, symptom_encoded], axis=1)  # (1, n_animal_features + n_symptom_features)
    
    # Ensure the input has the correct length (496 features in total)
    expected_input_length = len(animal_encoder.classes_) + len(symptom_encoder.classes_)
    
    if input_features.shape[1] != expected_input_length:
        # Pad or truncate the input_features to match the expected input length (496)
        input_features = np.pad(input_features, ((0, 0), (0, expected_input_length - input_features.shape[1])), mode='constant')
    
    # Predict the disease
    prediction = model.predict(input_features)
    
    # Decode the predicted disease
    predicted_disease = disease_encoder.inverse_transform([prediction.argmax()])[0]
    
    return predicted_disease

# Streamlit UI
st.title('Animal Disease Prediction')
st.write("Enter the animal name and symptom to predict the disease:")

# User inputs
#animal_name = st.('Animal Name', '')
animals = ["cow", "buffalo", "sheep", "goat"]

# Create a selectbox for animal input
animal_name = st.selectbox("Select an Animal", animals)
symptom_name = st.text_input('Symptom', '')

# Prediction on button click
if st.button('Predict'):
    if animal_name and symptom_name:
        result = predict_disease(animal_name, symptom_name)
        st.write(f"The predicted disease is: {result}")
    else:
        st.warning("Please enter both animal name and symptom.")
