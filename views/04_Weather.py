# Import necessary libraries
import streamlit as st
import requests

# Set your OpenWeather and OpenCage API keys
API_KEY = "6fa105bb23df102a78770e15ace55ffc"
OPENCAGE_API_KEY = "6fa2229fd48f4b7f8a8fef3d55bbc35d"

# Function to get latitude and longitude from city name
def get_lat_lon_from_city(city):
    geocode_url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key={OPENCAGE_API_KEY}"
    response = requests.get(geocode_url)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lon = data['results'][0]['geometry']['lng']
        return lat, lon
    else:
        return None, None

# Function to get current weather data
def get_current_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        rainfall = weather_data.get('rain', {}).get('3h', None)  # Extract rainfall data (if available)
        return weather_data, rainfall
    else:
        st.error(f"Error fetching weather data: {response.status_code}")
        return None, None

# Disease Mapping Based on Temperature, Humidity, and Rainfall
disease_mapping = {
    "hot_humid": {
        "Crops": ["Powdery Mildew", "Leaf Spot", "Downy Mildew", "Bacterial Blight"],
        "Livestock": ["Bovine Respiratory Disease", "Coccidiosis", "Heat Stress"]
    },
    "cold_humid": {
        "Crops": ["Late Blight", "Clubroot", "Root Rot", "Bacterial Wilt"],
        "Livestock": ["Pneumonia", "Foot Rot", "Blue Tongue Disease"]
    },
    "hot_dry": {
        "Crops": ["Drought Stress", "Leaf Curl Virus", "Spider Mites"],
        "Livestock": ["Heat Stress", "Tick-borne Diseases"]
    },
    "moderate_conditions": {
        "Crops": ["Gray Mold", "Alternaria Blight", "Rust"],
        "Livestock": ["Ringworm", "Pink Eye"]
    },
    "cold_dry": {
        "Crops": ["Frost Damage", "Seedling Blight"],
        "Livestock": ["Frostbite", "Respiratory Diseases"]
    },
    "high_rain_high_humidity": {
        "Crops": ["Bacterial Blight", "Root Rot"],
        "Livestock": ["Foot Rot"]
    },
    "low_rain_low_humidity": {
        "Crops": ["Drought Stress", "Leaf Curl Virus"],
        "Livestock": ["Heat Stress"]
    }
}

# Function to determine disease risks
def get_possible_diseases(temp, humidity, wind_speed, rainfall):
    if temp > 25 and humidity > 70:
        return disease_mapping["hot_humid"]
    elif temp < 20 and humidity > 80:
        return disease_mapping["cold_humid"]
    elif temp > 30 and humidity < 60:
        return disease_mapping["hot_dry"]
    elif 15 <= temp <= 25 and 50 <= humidity <= 70:
        return disease_mapping["moderate_conditions"]
    elif temp < 15 and humidity < 50:
        return disease_mapping["cold_dry"]
    elif rainfall > 50 and humidity > 70:
        return disease_mapping["high_rain_high_humidity"]
    elif rainfall < 20 and humidity < 60:
        return disease_mapping["low_rain_low_humidity"]
    else:
        return {"Crops": ["No major risks detected"], "Livestock": ["No major risks detected"]}

# Function to display weather and disease predictions
def display_weather_and_diseases(weather_data, rainfall, city):
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    wind_speed = weather_data['wind']['speed']

    st.write(f"### Current Weather in {city}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temperature (°C)", f"{temp}°C")
    with col2:
        st.metric("Humidity (%)", f"{humidity}%")
    with col3:
        st.metric("Wind Speed (m/s)", f"{wind_speed} m/s")
    with col4:
        if rainfall is not None:
            st.metric("Rainfall (mm)", f"{rainfall} mm (last 3 hours)")
        else:
            st.metric("Rainfall (mm)", "No Data")

    diseases = get_possible_diseases(temp, humidity, wind_speed, rainfall if rainfall else 0)
    
    st.write("### Potential Diseases Based on Current Conditions")
    colL,colC=st.columns(2)
    with colL:
        st.write("#### Livestock Diseases:")
        for disease in diseases["Livestock"]:
            st.write(f"- {disease}")
    with colC:
        st.write("#### Crop Diseases:")
        for disease in diseases["Crops"]:
            st.write(f"- {disease}")

# Streamlit app logic
st.title("Comprehensive Weather-Based Disease Predictor")

city = st.text_input("Enter your city:")
if st.button("Get Weather and Disease Predictions"):
    if city:
        lat, lon = get_lat_lon_from_city(city)
        if lat and lon:
            weather_data, rainfall = get_current_weather(lat, lon)
            if weather_data:
                display_weather_and_diseases(weather_data, rainfall, city)
            else:
                st.error("Unable to fetch weather data.")
        else:
            st.error("Invalid city. Please try again.")
    else:
        st.warning("Please enter a city.")
