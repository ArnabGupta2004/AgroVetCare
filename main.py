import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import firebase_setup  # Import the Firebase setup
from firebase_admin import auth, db
import requests  # For geocoding API

st.set_page_config(
    page_title="AgroVet Care",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")

# Define Pages
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
    title="Chatbox",
)
acc = st.Page(
    page="views/acc.py",
    title="Account",
)


# Initialize session state for authentication and location
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "location_collected" not in st.session_state:
    st.session_state.location_collected = False

# Geocode Function to Get Latitude and Longitude
def geocode_location(city, district):
    api_key = "6fa2229fd48f4b7f8a8fef3d55bbc35d"  # Replace with your API key
    location = f"{city}, {district}"
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            lat = data["results"][0]["geometry"]["lat"]
            lon = data["results"][0]["geometry"]["lng"]
            return lat, lon
        else:
            st.warning("Unable to find latitude and longitude for the given location.")
    else:
        st.error("Error connecting to the geocoding API.")
    return None, None

# Collect Location Data
def collect_location():
    st.header("Location Details")
    city = st.text_input("City", key="city_input")
    district = st.text_input("District", key="district_input")

    if st.button("Submit Location"):
        if city and district:
            # Get latitude and longitude from geocoding API
            lat, lon = geocode_location(city, district)

            if lat is not None and lon is not None:
                # Store the input along with latitude and longitude in Firebase
                location_data = {
                    "city": city,
                    "district": district,
                    "latitude": lat,
                    "longitude": lon
                }
                db.reference(f'users/{st.session_state.user_id}/location').set(location_data)

                # Update session state
                st.session_state.city = city
                st.session_state.district = district
                st.session_state.latitude = lat
                st.session_state.longitude = lon
                st.session_state.location_collected = True

                st.success(f"Location saved successfully! Latitude: {lat}, Longitude: {lon}")
                if st.button("Go to Home Page"):
                    pass
            else:
                st.warning("Could not fetch latitude and longitude for the given location.")
        else:
            st.warning("Please fill in both City and District.")

# Firebase Sign-In Function
def sign_in(email, password):
    try:
        user = auth.get_user_by_email(email)
        if password == db.reference(f'users/{user.uid}/password').get():
            return user.uid
        else:
            return None
    except Exception as e:
        st.error(f"Error during sign-in: {e}")
        return None

# Login Page Function
def login_page():
    st.title("Account")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        st.header("Sign In")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In"):
            user_id = sign_in(email, password)
            if user_id:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.success("Logged in successfully!")
                if st.button("Next"):
                    pass
            else:
                st.error("Invalid email or password.")

    with tab2:
        st.header("Sign Up")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            try:
                user = auth.create_user(email=new_email, password=new_password)
                user_data = {
                    "email": new_email,
                    "password": new_password,
                }
                db.reference(f'users/{user.uid}').set(user_data)
                st.success("Account created successfully! Please log in.")
            except Exception as e:
                st.error(f"Sign-Up failed: {e}")

# Main Application
if not st.session_state.logged_in:
    login_page()
elif not st.session_state.location_collected:
    collect_location()
else:
    # Navigation and app content
    pages = [home, edu, dr, cb, vet, about]
    app_mode = st.navigation(pages=pages)
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
