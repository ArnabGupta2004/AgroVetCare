#API_KEY = "6fa105bb23df102a78770e15ace55ffc"
import streamlit as st
import requests
import datetime

# OpenWeather API key
API_KEY = st.secrets["openweather"]["api_key"]

# Function to get user's current location (using ipinfo.io)
def get_location():
    res = requests.get('https://ipinfo.io')
    data = res.json()
    loc = data['loc'].split(',')
    return float(loc[0]), float(loc[1]), data['city']

# Function to get weather forecast
def get_weather_forecast(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error(f"Error fetching weather data: {response.status_code}")
        return None
    
    return response.json()

# Function to check for severe weather alerts

def check_for_severe_weather(forecast_data):
    severe_conditions = ['storm', 'rain', 'thunderstorm', 'hail', 'snow']
    alerts = []

    for entry in forecast_data['list']:
        weather_desc = entry['weather'][0]['description'].lower()
        if any(condition in weather_desc for condition in severe_conditions):
            date = datetime.datetime.fromtimestamp(entry['dt']).strftime('%Y-%m-%d %H:%M:%S')
            alerts.append(f"On {date}, severe weather condition detected: {weather_desc.title()}")

    return alerts

# Function to aggregate the 5-day 3-hour forecast into daily summaries
def aggregate_daily_forecast(forecast_data):
    # Aggregating data into daily summaries
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

# Display weather data
def display_forecast(forecast_data, city):
    if forecast_data is None:
        st.error("No forecast data available to display.")
        return

    # Check for severe weather alerts
    if(st.button("Upcoming Alerts")):
        alerts = check_for_severe_weather(forecast_data)
        
        if alerts:
            st.write("### Weather Alerts")
            for alert in alerts:
                st.write(alert)
        else:
            st.write("### No severe weather alerts in the forecast.")

    # Display 5-day weather forecast
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

# Streamlit app UI
st.title("Weather Alerts")

# Get the user's location
lat, lon, city = get_location()

# Get weather forecast data
forecast_data = get_weather_forecast(lat, lon)

# Display the forecast
display_forecast(forecast_data, city)
