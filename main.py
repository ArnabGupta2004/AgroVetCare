import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

# Define your pages
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

# Display logo and sidebar information
st.logo("AgroVet Care_logo.png")


# Feedback Section in Sidebar
st.sidebar.markdown("---")  # Separator
st.sidebar.subheader("We Value Your Feedback")
feedback = st.sidebar.text_area("Please provide your feedback below:")

if st.sidebar.button("Submit Feedback"):
    if feedback:
        st.sidebar.success("Thank you for your feedback!")
        # Here you can process the feedback, e.g., store it in a database or send via email
    else:
        st.sidebar.warning("Please enter your feedback before submitting.")

st.sidebar.text("Made by Team Code&Conquer")
