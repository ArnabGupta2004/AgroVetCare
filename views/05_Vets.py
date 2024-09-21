import streamlit as st
import folium
from folium import Marker
from streamlit_folium import folium_static
import requests

# Sample data for nearby vets and medicine shops
vets = [
    {"name": "Vet Clinic A", "location": (22.620462, 88.386270)},  # Example coordinates
    {"name": "Vet Clinic B", "location": (22.618006, 88.383436)},
]

shops = [
    {"name": "Medicine Shop A", "location": (22.620858, 88.383951)},
    {"name": "Medicine Shop B", "location": (22.615946, 88.386786)},
]


# Get user's location (replace with your preferred method)
# Example coordinates for demonstration
user_location = (22.619270, 88.383888)  # You can replace this with the user's actual location

# Create a map centered at the user's location
m = folium.Map(location=user_location, zoom_start=16)

# Add a marker for the user's location
folium.Marker(location=user_location, tooltip="You are here", icon=folium.Icon(color='blue')).add_to(m)

# Add markers for vets
for vet in vets:
    Marker(location=vet["location"], tooltip=vet["name"], icon=folium.Icon(color='red')).add_to(m)

# Add markers for medicine shops
for shop in shops:
    Marker(location=shop["location"], tooltip=shop["name"], icon=folium.Icon(color='green')).add_to(m)

# Get the screen width to adjust the map size dynamically for mobile responsiveness
if st.session_state.get("screen_width") is None:
    st.session_state.screen_width = 150  # Default width if we can't detect

# Input to dynamically change map width, detecting mobile screen size
map_width = 340
map_height = 500  # Set a fixed height

# Display the map with dynamic width for mobile responsiveness
st.header("Map of Nearby Vets and Medicine Shops")
folium_static(m, width=map_width, height=map_height)

st.write("🔵➜ Your Location 🔴➜ Vets 🟢➜ Medicines")

# Button to trigger the pop-up
if st.button("Add Your Store"):
    # Simulated pop-up using an expander
    with st.expander("Pop-up Window", expanded=True):
        st.write("Enter Store Details")
        st.text_input("Enter Store Name:")
        st.text_input("Enter Store Location:")
        st.button("Submit")
