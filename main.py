import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

# Load Custom CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the CSS
load_css("style.css")


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
    title="About Uss",
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


# Feedback Section in Sidebar with Animation
st.sidebar.subheader("We Value Your Feedback")
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
feedback = st.sidebar.text_area("Please provide your feedback below:")
if st.sidebar.button("Submit Feedback"):
    if feedback:
        st.sidebar.success("Thank you for your feedback!")
        # Here you can process the feedback, e.g., store it in a database or send via email
    else:
        st.sidebar.warning("Please enter your feedback before submitting.")
st.markdown('</div>', unsafe_allow_html=True)
