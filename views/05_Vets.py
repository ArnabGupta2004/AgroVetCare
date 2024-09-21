import streamlit as st
import folium
from folium import Marker
from streamlit_folium import folium_static
import requests

# Sample data for nearby vets and medicine shops
# Replace these with your actual data sources
vets = [
    {"name": "Vet Clinic A", "location": (28.7041, 77.1025)},  # Example coordinates
    {"name": "Vet Clinic B", "location": (28.5355, 77.3910)},
]

shops = [
    {"name": "Medicine Shop A", "location": (28.7041, 77.1250)},
    {"name": "Medicine Shop B", "location": (28.6519, 77.2213)},
]

# Streamlit UI
st.title("Nearby Vets and Medicine Shops")

# Get user's location (replace with your preferred method)
# Example coordinates for demonstration
user_location = (28.7041, 77.1025)  # You can replace this with the user's actual location

# Create a map centered at the user's location
m = folium.Map(location=user_location, zoom_start=12)

# Add a marker for the user's location
folium.Marker(location=user_location, tooltip="You are here", icon=folium.Icon(color='blue')).add_to(m)

# Add markers for vets
for vet in vets:
    Marker(location=vet["location"], tooltip=vet["name"], icon=folium.Icon(color='green')).add_to(m)

# Add markers for medicine shops
for shop in shops:
    Marker(location=shop["location"], tooltip=shop["name"], icon=folium.Icon(color='red')).add_to(m)

# Display the map
st.header("Map of Nearby Vets and Medicine Shops")
folium_static(m)



# Title
st.title("Simulated Pop-up in Streamlit")

# Button to trigger the pop-up
if st.button("Show Pop-up"):
    # Simulated pop-up using an expander
    with st.expander("Pop-up Window", expanded=True):
        st.write("This is the content of the pop-up window!")
        st.text_input("Enter some input:")
        st.button("Submit")
else:
    st.write("Press the button to open a pop-up.")
