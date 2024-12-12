import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import firebase_setup  # Import the Firebase setup
from firebase_admin import auth, db

st.set_page_config(
    page_title="AgroVet Care",       # Set the page title
    layout="centered",                   # Use wide layout
    initial_sidebar_state="collapsed"  # Sidebar starts collapsed (closed)
)


# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")

# Function to prepend country code
def format_phone_number(phone_number):
    # Assuming India (+91) for this example, modify as needed
    if phone_number.startswith("+"):
        return phone_number  # Already in E.164 format
    else:
        return "+91" + phone_number  # Prepend India country code (+91)

# Initialize session state for authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Firebase Sign-In Function (Phone and Password)
def sign_in(phone_number, password):
    try:
        formatted_phone = format_phone_number(phone_number)  # Format phone number
        user = auth.get_user_by_phone_number(formatted_phone)
        if password == db.reference(f'users/{user.uid}/password').get():  # Check the stored password
            return user.uid
        else:
            return None
    except Exception as e:
        st.error(f"Error during sign-in: {e}")
        return None

# Firebase Sign-Up Function (Phone and Password)
# Firebase Sign-Up Function (Phone and Password)
def sign_up(phone_number, password):
    try:
        formatted_phone = format_phone_number(phone_number)  # Format phone number
        # Create a new user with phone number and password
        user = auth.create_user(phone_number=formatted_phone)
        user_data = {
            "phone_number": formatted_phone,
            "password": password  # Avoid storing plain-text passwords in production
        }
        # Store the user details in Firebase Realtime Database
        db.reference(f'users/{user.uid}').set(user_data)
        return user.uid
    except Exception as e:
        st.error(f"Error during sign-up: {e}")
        return None


# Login Page Function
def login_page():
    st.title("Account")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        st.header("Sign In")
        phone_number = st.text_input("Phone Number", key="signin_phone_number", placeholder="Enter your phone number")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In"):
            user_id = sign_in(phone_number, password)
            if user_id:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.success("Logged in successfully!")
                if st.button("Next"):
                    pass
            else:
                st.error("Invalid Phone Number or password.")

    with tab2:
        st.header("Sign Up")
        new_phone_number = st.text_input("Phone Number", key="signup_phone_number", placeholder="Enter your phone number")
        new_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            user_id = sign_up(new_phone_number, new_password)
            if user_id:
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Sign-Up failed.")



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
acc = st.Page(
    page="views/acc.py",
    title="Account",
)

cb=st.Page(
    page="views/07_Chatbox.py",
    title="Text Analysis",
)

# Pages Navigation
pages = [home, dr, we, vet, cb]

# Display login page if not authenticated
if not st.session_state.logged_in:
    login_page()
else:
    app_mode = st.navigation(pages=pages)
    app_mode.run()

# Display logo and sidebar information
st.logo("AgroVet Care_logo.png")

# Feedback Section in Sidebar
if st.session_state.logged_in:
    st.sidebar.subheader(":mailbox: Get In Touch With Us!")
    contact_form = """
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
    feedback = st.sidebar.text_area("Please provide your feedback below:")

    if st.sidebar.button("Submit Feedback"):
        if feedback:
            st.sidebar.success("Thank you for your feedback!")
        else:
            st.sidebar.warning("Please enter your feedback before submitting.")
else:
    st.sidebar.info("Log in to submit feedback.")

st.sidebar.text("Made by Team Code&Conquer_TMSL")
