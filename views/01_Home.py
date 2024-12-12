import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import urllib.parse
import joblib
import json
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tempfile
from streamlit_lottie import st_lottie
from googletrans import Translator
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os

# Instantiate the translator object
translator = Translator()

col0,colt=st.columns([4,1],gap="large")
with col0:
    # Inject the CSS with st.markdown
    st.image("AgroVet Care_logo.png", use_column_width=True)

with colt:
    # Display the language selection dropdown
    languages = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'ur': 'Urdu'
    }

    selected_language = st.selectbox("Select Language", options=languages.keys(), format_func=lambda x: languages[x])

# Function to translate text
def translate_text(text, lang='en'):
        return text  # Return original text if translation fails

# Translate content based on selected language
#def translate_content():
    #st.markdown(f"<h2 style='text-align: center;'>{translate_text('Change Language', selected_language)}</h2>", unsafe_allow_html=True)
    #st.markdown(f"<h7 style='text-align: center;'>{translate_text('Welcome to the Disease Prediction System! 🌿🐄🔍', selected_language)}</h7>", unsafe_allow_html=True)



#translate_content()

st.markdown(f"<div style='text-align: center;'><h7>{translate_text('Welcome to the Disease Prediction System! 🌿🐄🔍', selected_language)}</h7></div>", unsafe_allow_html=True)

def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_working=load_lottiefile("lottiefiles/working.json")
lottie_crop=load_lottiefile("lottiefiles/crop.json")
lottie_cow=load_lottiefile("lottiefiles/cow.json")

# Define cure information with Google search links
def google_search_link(disease_name):
    query = urllib.parse.quote(disease_name + " cure")
    return f"https://www.google.com/search?q={query}"

# Update crop_cures with all class names
crop_cures = {
    'Apple___Apple_scab': google_search_link('Apple Apple scab cure'),
    'Apple___Black_rot': google_search_link('Apple Black rot cure'),
    'Apple___Cedar_apple_rust': google_search_link('Apple Cedar apple rust cure'),
    'Apple___healthy': google_search_link('Healthy Apple'),
    'Blueberry___healthy': google_search_link('Healthy Blueberry'),
    'Cherry_(including_sour)___Powdery_mildew': google_search_link('Cherry Powdery mildew cure'),
    'Cherry_(including_sour)___healthy': google_search_link('Healthy Cherry'),
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': google_search_link('Corn Cercospora leaf spot Gray leaf spot cure'),
    'Corn_(maize)___Common_rust_': google_search_link('Corn Common rust cure'),
    'Corn_(maize)___Northern_Leaf_Blight': google_search_link('Corn Northern Leaf Blight cure'),
    'Corn_(maize)___healthy': google_search_link('Healthy Corn'),
    'Grape___Black_rot': google_search_link('Grape Black rot cure'),
    'Grape___Esca_(Black_Measles)': google_search_link('Grape Esca Black Measles cure'),
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': google_search_link('Grape Leaf blight Isariopsis Leaf Spot cure'),
    'Grape___healthy': google_search_link('Healthy Grape'),
    'Orange___Haunglongbing_(Citrus_greening)': google_search_link('Orange Haunglongbing Citrus greening cure'),
    'Peach___Bacterial_spot': google_search_link('Peach Bacterial spot cure'),
    'Peach___healthy': google_search_link('Healthy Peach'),
    'Pepper,_bell___Bacterial_spot': google_search_link('Pepper bell Bacterial spot cure'),
    'Pepper,_bell___healthy': google_search_link('Healthy Pepper bell'),
    'Potato___Early_blight': google_search_link('Potato Early blight cure'),
    'Potato___Late_blight': google_search_link('Potato Late blight cure'),
    'Potato___healthy': google_search_link('Healthy Potato'),
    'Raspberry___healthy': google_search_link('Healthy Raspberry'),
    'Soybean___healthy': google_search_link('Healthy Soybean'),
    'Squash___Powdery_mildew': google_search_link('Squash Powdery mildew cure'),
    'Strawberry___Leaf_scorch': google_search_link('Strawberry Leaf scorch cure'),
    'Strawberry___healthy': google_search_link('Healthy Strawberry'),
    'Tomato___Bacterial_spot': google_search_link('Tomato Bacterial spot cure'),
    'Tomato___Early_blight': google_search_link('Tomato Early blight cure'),
    'Tomato___Late_blight': google_search_link('Tomato Late blight cure'),
    'Tomato___Leaf_Mold': google_search_link('Tomato Leaf Mold cure'),
    'Tomato___Septoria_leaf_spot': google_search_link('Tomato Septoria leaf spot cure'),
    'Tomato___Spider_mites Two-spotted_spider_mite': google_search_link('Tomato Spider mites Two-spotted spider mite cure'),
    'Tomato___Target_Spot': google_search_link('Tomato Target Spot cure'),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': google_search_link('Tomato Tomato Yellow Leaf Curl Virus cure'),
    'Tomato___Tomato_mosaic_virus': google_search_link('Tomato Tomato mosaic virus cure'),
    'Tomato___healthy': google_search_link('Healthy Tomato')
}

livestock_cures = {
    '(BRD) Bovine Dermatitis Disease healthy lumpy': google_search_link('(BRD) Bovine Dermatitis Disease healthy lumpy cure'),
    '(BRD) Bovine Disease Respiratory': google_search_link('(BRD) Bovine Disease Respiratory cure'),
    'Contagious Ecthym': google_search_link('Contagious Ecthym cure'),
    'Dermatitis': google_search_link('Dermatitis cure'),
    'healthy': google_search_link('Healthy'),
    'healthy lumpy skin': google_search_link('healthy lumpy skin cure'),
    'lumpy skin': google_search_link('lumpy skin cure')
}

cattle_names=['Foot and Mouth disease','Healthy','Lumpy Skin Disease']

poultry_names=['cocci','healthy','ncd','salmo']

pig_names=['Healthy','Infected_Bacterial_Erysipelas','Infected_Bacterial_Greasy_Pig_Disease','Infected_Environmental_Dermatitis','Infected_Environmental_Sunburn','Infected_Fungal_Pityriasis_Rosea','Infected_Fungal_Ringworm','Infected_Parasitic_Mange','Infected_Viral_Foot_and_Mouth_Disease','Infected_Viral_Swinepox']

goat_names=['Boqueira','Mal do caroco']

bee_names=['ant_problems','few_varrao_and_hive_beetles','healthy','hive_being_robbed','missing_queen','varroa_and_small_hive_beetles']

def classify_image(uploaded_file, green_threshold=15):
    import cv2
    import numpy as np
    from skimage.feature import local_binary_pattern

    # Save uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and process the image
    image = cv2.imread(temp_file_path)
    if image is None:
        raise ValueError("Could not read the image. Ensure the uploaded file is an image.")

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green mapping
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

    # Calculate green pixel percentage
    green_pixels = cv2.countNonZero(green_mask)
    total_pixels = image.shape[0] * image.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100

    # If green percentage is below threshold, classify as "Not Plant"
    if green_percentage < green_threshold:
        return "Not Plant"

    # Texture analysis on green regions
    green_regions = cv2.bitwise_and(image, image, mask=green_mask)
    gray_green = cv2.cvtColor(green_regions, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_green, P=8, R=1, method="uniform")

    # Use texture features for classification (dummy logic here)
    texture_score = np.mean(lbp)
    classification = 1 if texture_score > 5 else 0

    return classification  

def crop_model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)  # index of the predicted class
    confidence = predictions[0][predicted_index] * 100  # confidence in percentage
    return predicted_index, confidence

def livestock_model_prediction(test_image):
    model = tf.keras.models.load_model("trained_livestock_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

def cattle_model_prediction(test_image):
    # Load the pre-trained model
    model = tf.keras.models.load_model("cattle_v1.h5")
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    # Make predictions
    predictions = model.predict(input_arr)
    # Get the index of the maximum probability and the corresponding confidence
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    return predicted_index, confidence  # Return both the index and confidence

def poultry_model_prediction(test_image):
    # Load the pre-trained model
    model = tf.keras.models.load_model("poultry_v1.h5")
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    # Make predictions
    predictions = model.predict(input_arr)
    # Get the index of the maximum probability and the corresponding confidence
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    return predicted_index, confidence  # Return both the index and confidence


# Disease recognition sections
st.markdown(f"<div style='text-align: center;'><h1>{translate_text('Disease Recognition', selected_language)}</h1></div>", unsafe_allow_html=True)


dr_ch = option_menu(
    menu_title=None,
    options=[translate_text("LiveStock", selected_language), translate_text("Crop", selected_language)],
    icons=["square-fill", "square-fill"],
    default_index=0,
    orientation="horizontal"
)

# Load the saved model
text_model = load_model('disease_prediction_model_2.h5')

# Load the encoders for animal and symptom
animal_encoder = joblib.load('animal_encoder.pkl')  # Save your encoder with joblib
symptom_encoder = joblib.load('symptom_encoder.pkl')  # Save your encoder with joblib
disease_encoder = joblib.load('disease_encoder.pkl')  # Save your encoder with joblib


# Define the prediction function
def predict_disease(animal_name, symptom_name):
    try:
        # Check if animal_name exists in encoder classes
        if animal_name in animal_encoder.classes_:
            animal_encoded = animal_encoder.transform([animal_name]).reshape(1, -1)
        else:
            # Handle unseen animal by using a zero-vector with the correct length
            animal_encoded = np.zeros((1, len(animal_encoder.classes_)))  # Zero-vector for unknown category
            st.warning(f"Unknown animal: {animal_name}. Using default encoding.")
    except Exception as e:
        st.error(f"Error processing animal: {e}")
        animal_encoded = np.zeros((1, len(animal_encoder.classes_)))

    try:
        # Check if symptom_name exists in encoder classes
        if symptom_name in symptom_encoder.classes_:
            symptom_encoded = symptom_encoder.transform([symptom_name]).reshape(1, -1)
        else:
            # Handle unseen symptom by using a zero-vector with the correct length
            symptom_encoded = np.zeros((1, len(symptom_encoder.classes_)))  # Zero-vector for unknown category
            #st.warning(f"Unknown symptom: {symptom_name}. Using default encoding.")
    except Exception as e:
        st.error(f"Error processing symptom: {e}")
        symptom_encoded = np.zeros((1, len(symptom_encoder.classes_)))

    # Combine the encoded features (ensure both are 2D arrays and match the expected length)
    input_features = np.concatenate([animal_encoded, symptom_encoded], axis=1)  # (1, n_animal_features + n_symptom_features)

    # Ensure the input has the correct length (496 features in total)
    expected_input_length = len(animal_encoder.classes_) + len(symptom_encoder.classes_)

    if input_features.shape[1] != expected_input_length:
        # Pad or truncate the input_features to match the expected input length (496)
        input_features = np.pad(input_features, ((0, 0), (0, expected_input_length - input_features.shape[1])), mode='constant')

    # Predict the disease probabilities
    prediction_probabilities = text_model.predict(input_features)[0]

    # Get the top 3-5 diseases with their probabilities
    top_indices = np.argsort(prediction_probabilities)[::-1][:5]
    top_diseases = disease_encoder.inverse_transform(top_indices)
    top_probabilities = prediction_probabilities[top_indices]

    # Prepare the predictions to display
    predictions = []
    for disease, prob in zip(top_diseases, top_probabilities):
        probability_percentage = f"{prob * 100:.2f}%"
        if prob == 0:
            probability_percentage = "<10%"  # If the probability is 0, display "<10%"
        predictions.append(f"{disease}: {probability_percentage}")

    # Get the disease with the highest probability
    max_prob_idx = top_indices[0]
    predicted_disease = disease_encoder.inverse_transform([max_prob_idx])[0]

    return predictions, predicted_disease


# Function to log input and output to an Excel file
def log_data(animal_name, symptom_name, disease):
    # Define the file name
    file_name = 'disease_predictions_log.xlsx'

    # Create a dictionary for the data
    data = {
        'Animal': [animal_name],
        'Symptom': [symptom_name],
        'Disease': [disease]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Check if the file exists
    if os.path.exists(file_name):
        # Append the data to the existing file
        existing_data = pd.read_excel(file_name)
        updated_data = pd.concat([existing_data, df], ignore_index=True)
        updated_data.to_excel(file_name, index=False)
    else:
        # Create a new file and save the data
        df.to_excel(file_name, index=False)



if dr_ch == translate_text("LiveStock", selected_language):

    st.header(translate_text("Livestock Disease Recognition", selected_language))
    st.write(translate_text("Choose Category:", selected_language))

    # Selectbox for category selection with Cattle as the default option
    category = st.selectbox(
        translate_text("Select Livestock Category:", selected_language),
        [
            translate_text("Cattle", selected_language),
            translate_text("Poultry", selected_language)
        ],
        index=0  # Set Cattle as the default option
    )
    
    test_images = []  # Initialize the list to store the uploaded files

    tab1, tab2, tab3, tab4 = st.tabs([translate_text("Text Input", selected_language),translate_text("Upload Images", selected_language), translate_text("Capture from Camera", selected_language),translate_text("Text + Image", selected_language)])
    
    with tab1:
        # Streamlit UI
        st.title(translate_text('Animal Disease Prediction', selected_language))
        st.write(translate_text("Enter the animal name and symptom to predict the disease:", selected_language))

        # User inputs
        animals = ["cow", "buffalo", "sheep", "goat"]

        # Create a selectbox for animal input
        any_animal_name = st.selectbox(translate_text("Select an Animal", selected_language), animals)
        any_symptom_name = st.text_input(translate_text('Symptom', selected_language), '')

        animal_name = translate_text(any_animal_name, 'en')
        symptom_name = translate_text(any_symptom_name, 'en')

        # Prediction on button click
        if st.button(translate_text('Predict', selected_language)):
            if animal_name and symptom_name:
                predictions, predicted_disease = predict_disease(animal_name, symptom_name)

                st.write(translate_text("Possible diseases:", selected_language))
                for prediction in predictions:
                    st.write(prediction)

                # Log the data to the Excel file
                log_data(animal_name, symptom_name, predicted_disease)

                with st.expander(translate_text("Know More", selected_language)):
                    st.markdown("[Know more about your disease](https://agrovetcare-yqz3vvwra2bveydzyzqlsq.streamlit.app/Education)")

                with st.expander(translate_text("Visit Marketplace", selected_language)):
                    st.markdown("[Visit Amazon Marketplace](https://www.amazon.in)")

                with st.expander(translate_text("Contact Experts", selected_language)):
                    # Expert 1
                    col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                    with col1:
                        st.markdown(f"""
                        *Name*: Dr. Singh  
                        *Contact*: [9876543211](tel:9876543211)  
                        *Status*: :green[Online]  
                        """)
                        contact_form = f"""
                        <form action="https://formsubmit.co/arnabgupta983@gmail.com" method="POST">
                            <input type="hidden" name="_captcha" value="false">
                            <input type="text" name="animal" value="{animal_name}" placeholder="{translate_text('Animal Name', selected_language)}" required>
                            <input type="text" name="symptom" value="{symptom_name}" placeholder="{translate_text('Symptom', selected_language)}" required>
                            <textarea name="message" placeholder="{translate_text('Tell your problem', selected_language)}"></textarea>
                            <button type="submit">{translate_text('Send', selected_language)}</button>
                        </form>
                        """
                        st.markdown(contact_form, unsafe_allow_html=True)

                    with col2:
                        st.image("manavatar.png", width=50)  # Adjust the width and image path

                    st.markdown("---")  # Horizontal separator

                    # Expert 2
                    col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                    with col1:
                        st.markdown(f"""
                            *Name*: Dr. Sharma  
                            *Contact*: [1234567899](tel:1234567899)  
                            *Status*: :red[Offline]  
                            """)
                        contact_form = f"""
                        <form action="https://formsubmit.co/arnabgupta983@gmail.com" method="POST">
                            <input type="hidden" name="_captcha" value="false">
                            <input type="text" name="animal" value="{animal_name}" placeholder="{translate_text('Animal Name', selected_language)}" required>
                            <input type="text" name="symptom" value="{symptom_name}" placeholder="{translate_text('Symptom', selected_language)}" required>
                            <textarea name="message" placeholder="{translate_text('Tell your problem', selected_language)}"></textarea>
                            <button type="submit">{translate_text('Send', selected_language)}</button>
                        </form>
                        """
                        st.markdown(contact_form, unsafe_allow_html=True)

                    with col2:
                        st.image("womanavatar.png", width=50)  # Adjust the width and image path

            else:
                st.warning(translate_text("Please enter both animal name and symptom.", selected_language))
    # Tab 1: File Uploader (allows multiple file uploads)
    with tab2:
        test_images = st.file_uploader(translate_text("Choose Images:", selected_language), type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    # Tab 2: Camera Input
    with tab3:
        st.write(translate_text("You can capture up to 5 images.", selected_language))
        captured_images = []
        for i in range(5):
            st.write(translate_text(f"Capture Image {i + 1}:", selected_language))
            captured_file = st.camera_input(translate_text("Capture Image", selected_language), key=f"camera_input_{i}")
            if captured_file:
                captured_images.append(captured_file)
                if len(captured_images) < 5:
                    st.write(translate_text("Capture another image or proceed.", selected_language))
            else:
                break
        test_images.extend(captured_images)  # Add captured images to the test_images list



    if test_images:
        if len(test_images) > 5:
            st.error(translate_text("You can upload up to 5 images only.", selected_language))
            test_images = test_images[:5]  # Limit to 5 images
        
        for img in test_images:
            st.image(img, width=200)

        if st.button(translate_text("Predict for All Images", selected_language)):
            with st.spinner(translate_text("Please Wait....", selected_language)):
                predicted_indices = []  # Store predicted indices for all images

                for img in test_images:
                    if category == translate_text("Cattle", selected_language):
                        result_probs, _ = cattle_model_prediction(img)
                        predicted_indices.append(np.argmax(result_probs))
                    elif category == translate_text("Poultry", selected_language):
                        result_probs, _ = poultry_model_prediction(img)
                        predicted_indices.append(np.argmax(result_probs))

                # Determine the most frequent predicted index
                most_frequent_index = max(set(predicted_indices), key=predicted_indices.count)

                # Determine the predicted disease
                if category == translate_text("Cattle", selected_language):
                    predicted_disease = cattle_names[most_frequent_index]
                elif category == translate_text("Poultry", selected_language):
                    predicted_disease = poultry_names[most_frequent_index]

                st.success(f"""{translate_text("Model is predicting it's a", selected_language)} **{predicted_disease}** based on the most frequent prediction.""")
                
                # Additional buttons and information
                with st.expander(translate_text("Know More", selected_language)):
                    st.markdown("[Know more about your disease](https://agrovetcare-yqz3vvwra2bveydzyzqlsq.streamlit.app/Education)")
                    
                with st.expander(translate_text("Visit Marketplace", selected_language)):
                    st.markdown("[Visit Amazon Marketplace](https://www.amazon.in)")

                with st.expander(translate_text("Contact Experts", selected_language)):
                    # Expert 1
                    col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                    with col1:
                        st.markdown(f"""
                        **Name**: Dr. Singh  
                        **Contact**: [9876543211](tel:9876543211)  
                        **Status**: :green[Online]  
                        """)
                    with col2:
                        st.image("manavatar.png", width=50)  # Adjust the width and image path
                    
                    st.markdown("---")  # Horizontal separator
                    
                    # Expert 2
                    col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                    with col1:
                        st.markdown(f"""
                        **Name**: Dr. Sharma  
                        **Contact**: [1234567899](tel:1234567899)  
                        **Status**: :red[Offline]  
                        """)
                    with col2:
                        st.image("womanavatar.png", width=50)  # Adjust the width and image path

# Crop Disease Recognition Tab
if dr_ch == translate_text("Crop", selected_language):
    
    st.header(translate_text("Crop Disease Recognition", selected_language))
    test_image = None  # Initialize the variable to store the uploaded or captured file

    tab1, tab2 = st.tabs([translate_text("Upload Image", selected_language), translate_text("Capture from Camera", selected_language)])
    # Tab 1: File Uploader
    with tab1:
        test_image = st.file_uploader(translate_text("Choose an Image:", selected_language), type=["png", "jpg", "jpeg"])
    
    # Tab 2: Camera Input
    with tab2:
        captured_file = st.camera_input(translate_text("Capture Image:", selected_language))
    
    # Check if a file was uploaded from either tab
    if captured_file:
        test_image = captured_file 

    if test_image:
        st.image(test_image, width=200)
        if st.button(translate_text("Predict", selected_language)):
            with st.spinner(translate_text("Please Wait....", selected_language)):
                yn=classify_image(test_image)
                if yn==1:
                    result_index, confidence = crop_model_prediction(test_image)
                    class_names = list(crop_cures.keys())
                    predicted_disease = class_names[result_index]
                    
                    # Check if predicted_disease is in crop_cures
                    if predicted_disease in crop_cures:
                        cure_link = crop_cures[predicted_disease]
                        st.success(f"""{translate_text("Model is predicting it's a", selected_language)} **{predicted_disease}**.""")
                        st.markdown(f"[{translate_text('Find Cure for', selected_language)} {predicted_disease}]({cure_link})")
                        
                        # Additional buttons
                        with st.expander(translate_text("Visit Marketplace", selected_language)):
                            st.markdown("[Visit Amazon Marketplace](https://www.amazon.in)")

                        with st.expander(translate_text("Contact Experts", selected_language)):
                            
                            # Expert 1
                            col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                            with col1:
                                st.markdown(f"""
                                **Name**: Abc  
                                **Contact**: [9876543211](tel:9876543211)  
                                **Status**: :green[Online]  
                                """)
                            with col2:
                                st.image("manavatar.png", width=50)  # Adjust the width and image path
                            
                            st.markdown("---")  # Horizontal separator
                            
                            # Expert 2
                            col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                            with col1:
                                st.markdown(f"""
                                **Name**: Xyz  
                                **Contact**: [1234567899](tel:1234567899)  
                                **Status**: :red[Offline]  
                                """)
                            with col2:
                                st.image("womanavatar.png", width=50)  # Adjust the width and image path
                    else:
                        st.error(f"Prediction '{predicted_disease}' is not found in the cure dictionary.")
                else:
                    st.warning(translate_text("Uploaded image isn't a plant/ Upload better detailed image of diseased plant.", selected_language))



st.markdown("---")

col1, col2 = st.columns([2, 1], gap="small")
with col1:
    st.markdown(f"""
    ### {translate_text('How It Works', selected_language)}
    1. **{translate_text('Upload Image', selected_language)}:** {translate_text('Go to the Disease Prediction page and upload an image of a plant or animal with suspected diseases.', selected_language)}
    2. **{translate_text('Analysis', selected_language)}:** {translate_text('Our system will process the image using advanced AI algorithms to identify potential diseases.', selected_language)}
    3. **{translate_text('Results', selected_language)}:** {translate_text('View the analysis results and receive recommendations for treatment and further action.', selected_language)}
    """)

with col2:
    st_lottie(
        lottie_working,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height=250,
        width=250,
        key=None,
    )

st.markdown("---")

col3, col4 = st.columns([2, 1], gap="small")
with col3:
    st.markdown(f"""
    ### {translate_text('Crop Disease Prediction 🌿', selected_language)}  
    {translate_text('Our system leverages advanced AI models to detect diseases in a wide range of crops, including fruits, vegetables, and grains. Simply upload an image of the affected plant, and our system will analyze it to identify potential issues like fungal infections, bacterial diseases, or nutrient deficiencies. With accurate and fast predictions, you can take timely action to protect your crops and maximize your yield.', selected_language)}
    """)

with col4:
    st_lottie(
        lottie_crop,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height=250,
        width=250,
        key=None,
    )
    
st.markdown("---")

col5, col6 = st.columns([2, 1], gap="small")
with col5:
    st.markdown(f"""
    ### {translate_text('Livestocks Disease Prediction 🐄', selected_language)}  
    {translate_text('Keeping your livestock healthy is crucial for a thriving farm. Our system can identify common diseases in cattle, sheep, and other animals by analyzing uploaded images. From skin infections to respiratory issues, we provide accurate insights and treatment recommendations, helping you ensure the well-being of your animals and maintain a productive herd.', selected_language)}
    """)

with col6:
    st_lottie(
        lottie_cow,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height=250,
        width=250,
        key=None,
    )

st.markdown("---")
