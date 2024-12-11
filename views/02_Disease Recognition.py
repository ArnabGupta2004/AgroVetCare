import os
import streamlit as st

# Print the current working directory and list all files
st.write("Current Working Directory:", os.getcwd())
st.write("Directory Contents:", os.listdir(os.getcwd()))

# Check if the model file exists
model_path = os.path.join(os.getcwd(), "trained_plant_disease_model.keras")
st.write("Model Path:", model_path)
st.write("File Exists:", os.path.exists(model_path))
