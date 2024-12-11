import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="AgroVet Care",       # Set the page title
    layout="centered",                   # Use wide layout
    initial_sidebar_state="collapsed"  # Sidebar starts collapsed (closed)
)


# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles\style.css")

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

edu=st.Page(
    page="views/06_Education.py",
    title="FarmHelp",
)

cb=st.Page(
    page="views/07_Chatbox.py",
    title="Text Analysis",
)

app_mode= st.navigation(pages=[home,edu,dr,cb,about])
app_mode.run()
st.logo("AgroVet Care_logo.png")

st.sidebar.subheader(":mailbox: Get In Touch With Us!")
contact_form="""
<form action="https://formsubmit.co/codeandconquer26@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your Name" required>
     <input type="email" name="email" placeholder="Your Email" required>
     <textarea name="message" placeholder="Give Feedback"></textarea>
     <button type="submit">Send</button>
</form>
"""

st.sidebar.markdown(contact_form, unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.text("Made by Team Code&Conquer_TMSL")
