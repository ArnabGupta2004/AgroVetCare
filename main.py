import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

home=st.Page(
    page="views/01_Home.py",
    title="Home",
    default=True
)
dr=st.Page(
    page="views/02_Disease Recognition.py",
    title="Disease Recognition",
)
we=st.Page(
    page="views/04_Weather.py",
    title="Weather Alerts",
)
about=st.Page(
    page="views/03_About Us.py",
    title="About Us",
)
vet=st.Page(
    page="views/05_Vets.py",
    title="Nearby Vets",
)

app_mode= st.navigation(pages=[home,dr,we,vet,about])
app_mode.run()
st.logo("AgroVet Care_logo.png")
st.sidebar.text("Made by Team Code&Conquer")

#app_mode = st.sidebar.radio("",options=("Home","About","Disease Recognition"),index=0)
