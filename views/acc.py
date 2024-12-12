import streamlit as st

# Simulated database (in memory)
if "users" not in st.session_state:
    st.session_state.users = {"admin": "password"}  # Predefined user

# Initialize session state for user authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

def login_page():
    st.title("Account")
    
    # Choose between Sign In and Sign Up
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        st.header("Sign In")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.header("Sign Up")
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            if new_username in st.session_state.users:
                st.warning("Username already exists. Please choose another.")
            elif new_username and new_password:
                st.session_state.users[new_username] = new_password
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Please fill out both fields.")
    
    # Log out option for signed-in users
    if st.session_state.logged_in:
        st.success(f"Logged in as: {st.session_state.username}")
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.experimental_rerun()
