import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

# Define the different views
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

# Theme options
theme = st.sidebar.selectbox("Select Theme", options=["Default", "Light", "Dark"])

# Apply custom theme CSS
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #111;
            color: #eee;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
elif theme == "Light":
    st.markdown(
        """
        <style>
        body {
            background-color: #fff;
            color: #000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Navigation between pages
app_mode = st.navigation(pages=[home, dr, we, vet, about])
app_mode.run()

# Display logo and sidebar information
st.logo("AgroVet Care_logo.png")
st.sidebar.text("Made by Team Code&Conquer")
