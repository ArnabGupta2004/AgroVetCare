import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase with the credentials JSON file
cred = credentials.Certificate("path_to_your_firebase_credentials.json")
firebase_admin.initialize_app(cred)

# Simulated database (in memory)
if "users" not in st.session_state:
    st.session_state.users = {}

# Initialize session state for user authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "phone_number" not in st.session_state:
    st.session_state.phone_number = None

def login_page():
    st.title("Account")
    
    # Choose between Sign In and Sign Up
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        st.header("Sign In")
        phone_number = st.text_input("Phone Number", key="login_phone_number", placeholder="Enter your phone number")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Sign In"):
            try:
                # Use Firebase Authentication to sign in the user with phone number and password
                user = auth.get_user_by_phone_number(phone_number)
                if user and user.password == password:
                    st.session_state.logged_in = True
                    st.session_state.phone_number = phone_number
                    st.success(f"Welcome, {phone_number}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid phone number or password.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab2:
        st.header("Sign Up")
        new_phone_number = st.text_input("New Phone Number", key="signup_phone_number", placeholder="Enter your phone number")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        
        if st.button("Sign Up"):
            try:
                # Check if phone number already exists in Firebase
                user = auth.get_user_by_phone_number(new_phone_number)
                if user:
                    st.warning("Phone number already exists. Please choose another.")
                else:
                    # Create a new user with phone number and password
                    auth.create_user(
                        phone_number=new_phone_number,
                        password=new_password
                    )
                    st.session_state.users[new_phone_number] = new_password
                    st.success("Account created successfully! Please log in.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Log out option for signed-in users
    if st.session_state.logged_in:
        st.success(f"Logged in as: {st.session_state.phone_number}")
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.phone_number = None
            st.experimental_rerun()

# Call the login_page function to render the page
login_page()
