# Import necessary libraries
import streamlit as st
import requests
import datetime

# Set your OpenWeather and OpenCage API keys
API_KEY = st.secrets["openweather"]["api_key"]
OPENCAGE_API_KEY = st.secrets["opencage"]["api_key"]

# Function to get user's latitude, longitude, and city based on IP
def get_location():
    try:
        res = requests.get('https://ipinfo.io')
        data = res.json()
        loc = data['loc'].split(',')
        latitude, longitude = float(loc[0]), float(loc[1])
        city = data.get('city', 'Unknown')
        return latitude, longitude, city
    except Exception as e:
        st.error("Could not determine location.")
        return None, None, None

# Function to get current weather data for given coordinates
def get_current_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching data from OpenWeather API")
        return None

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

# Function to get weather forecast for given coordinates
def get_weather_forecast(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching weather data: {response.status_code}")
        return None
    return response.json()

# Function to aggregate 5-day 3-hour forecast data into daily summaries
def aggregate_daily_forecast(forecast_data):
    daily_forecast = {}
    
    for entry in forecast_data['list']:
        date = datetime.datetime.fromtimestamp(entry['dt']).strftime('%Y-%m-%d')
        temp = entry['main']['temp']
        weather_desc = entry['weather'][0]['description'].title()
        icon_code = entry['weather'][0]['icon']

        if date not in daily_forecast:
            daily_forecast[date] = {
                'temps': [temp],
                'weather_desc': weather_desc,
                'icon_code': icon_code
            }
        else:
            daily_forecast[date]['temps'].append(temp)

    # Calculate average temperature for each day
    for date, data in daily_forecast.items():
        data['avg_temp'] = sum(data['temps']) / len(data['temps'])
    
    return daily_forecast

# Function to check for severe weather alerts
def check_for_severe_weather(forecast_data):
    severe_conditions = ['storm', 'rain', 'thunderstorm', 'hail', 'snow']
    alerts = []

    for entry in forecast_data['list']:
        weather_desc = entry['weather'][0]['description'].lower()
        if any(condition in weather_desc for condition in severe_conditions):
            date = datetime.datetime.fromtimestamp(entry['dt']).strftime('%d-%m-%Y %H:%M:%S')
            alerts.append(f"On {date}: {weather_desc.title()}")

    return alerts

# Function to display current weather in metric columns
def display_current_weather(lat, lon, city):
    data = get_current_weather(lat, lon)
    if data:
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        st.write(f"### Current Weather for {city}")
        
        # Display metrics in three columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temperature (°C)", f"{temperature}°C")
        with col2:
            st.metric("Humidity (%)", f"{humidity}%")
        with col3:
            st.metric("Wind Speed (m/s)", f"{wind_speed} m/s")

# Function to display weather forecast and alerts
def display_forecast(forecast_data, city):
    if forecast_data is None:
        st.error("No forecast data available to display.")
        return

    # Display severe weather alerts
    with st.expander("Upcoming Alerts"):
        alerts = check_for_severe_weather(forecast_data)
        if alerts:
            st.write("### Weather Alerts")
            for alert in alerts:
                st.write(alert)
        else:
            st.write("### No severe weather alerts in the forecast.")

    # Aggregate forecast data into daily summaries
    daily_forecast = aggregate_daily_forecast(forecast_data)
    st.write(f"### 5-day Weather Forecast for {city}")
    
    for date, data in daily_forecast.items():
        avg_temp = data['avg_temp']
        weather_desc = data['weather_desc']
        icon_code = data['icon_code']
        icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png"
        
        st.write(f"**{date}**")
        st.write(f"Average Temperature: {avg_temp:.2f}°C")
        st.write(f"Weather: {weather_desc}")
        st.image(icon_url)

# Main Streamlit app UI
st.title("Weather Alerts")

# Automatic location-based current weather display
latitude, longitude, city = get_location()
if latitude and longitude:
    display_current_weather(latitude, longitude, city)

# User input for city to view forecast
st.header("Alerts")
city = st.text_input("Enter your city:")

if st.button("Get Weather Forecast"):
    if city:
        lat, lon = get_lat_lon_from_city(city)
        if lat and lon:
            forecast_data = get_weather_forecast(lat, lon)
            display_forecast(forecast_data, city)
        else:
            st.error("Could not find location. Please enter a valid city.")
    else:
        st.warning("Please enter a city name.")
