import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

# Define pages for navigation
home = st.Page(
    page="views/01_Home.py",
    title="Home",
    default=True
)
dr = st.Page(
    page="views/02_Disease Recognition.py",
    title="Disease Recognition",
)
we = st.Page(
    page="views/04_Weather.py",
    title="Weather Alerts",
)
about = st.Page(
    page="views/03_About Us.py",
    title="About Us",
)
vet = st.Page(
    page="views/05_Vets.py",
    title="Nearby Vets",
)

# Navigation menu
app_mode = st.navigation(pages=[home, dr, we, vet, about])
app_mode.run()

# Display logo and sidebar text
st.logo("AgroVet Care_logo.png")
st.sidebar.text("Made by Team Code&Conquer")

# Feedback section at the bottom
st.markdown("---")  # Adds a horizontal line for separation
st.header("Feedback")
feedback = st.text_area("Please share your feedback or suggestions here:")

# Feedback submit button
if st.button("Submit Feedback"):
    if feedback:
        # Handle feedback submission (e.g., save it to a file, database, or show a thank-you message)
        st.success("Thank you for your feedback!")
    else:
        st.error("Please enter your feedback before submitting.")
