import os
import streamlit as st

# Streamlit UI
st.title("Current Working Directory Checker")

# Get and display the current working directory
current_directory = os.getcwd()
st.write("Current working directory:", current_directory)
