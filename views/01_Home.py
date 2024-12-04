import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import urllib.parse
import joblib
import json
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import tempfile
from streamlit_lottie import st_lottie
from googletrans import Translator


# Instantiate the translator object
translator = Translator()

col0,colt=st.columns([4,1],gap="large")
with col0:
    # Inject the CSS with st.markdown
    st.image("AgroVet Care_logo.png", use_column_width=True)

with colt:
    # Display the language selection dropdown
    languages = ['en', 'hi', 'mr', 'ta', 'te', 'bn', 'ur']
    selected_language = st.selectbox("Select Language", languages)

# Function to translate text
def translate_text(text, lang='en'):
    try:
        translated = translator.translate(text, dest=lang)
        return translated.text
    except Exception as e:
        return text  # Return the original text if translation fails

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

goat_names=[' boqueira',' mal do caroco']

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
    model = tf.keras.models.load_model("trained_livestock_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

def cattle_model_prediction(test_image):
    model = tf.keras.models.load_model("cattle_v1.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

def poultry_model_prediction(test_image):
    model = tf.keras.models.load_model("poultry_v1.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

def pig_model_prediction(test_image):
    model = tf.keras.models.load_model("pig_v2.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

def goat_model_prediction(test_image):
    model = tf.keras.models.load_model("goat_v1.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

def bee_model_prediction(test_image):
    model = tf.keras.models.load_model("bee_v1.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Disease recognition sections
st.markdown(f"<div style='text-align: center;'><h1>{translate_text('Disease Recognition', selected_language)}</h1></div>", unsafe_allow_html=True)


dr_ch = option_menu(
    menu_title=None,
    options=[translate_text("LiveStock", selected_language), translate_text("Crop", selected_language)],
    icons=["square-fill", "square-fill"],
    default_index=0,
    orientation="horizontal"
)

if dr_ch == translate_text("LiveStock", selected_language):

    st.header(translate_text("Livestock Disease Recognition", selected_language))
    st.write(translate_text("Choose Category:", selected_language))

    # Selectbox for category selection with Cattle as the default option
    category = st.selectbox(
        translate_text("Select Livestock Category:", selected_language),
        [
            translate_text("Cattle", selected_language),
            translate_text("Poultry", selected_language),
            translate_text("Pig", selected_language),
            translate_text("Goat", selected_language),
            translate_text("Bees", selected_language)
        ],
        index=0  # Set Cattle as the default option
    )
    
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
                #yn = classify_image(test_image)
                yn=1
                if yn == 1:
                    result_index = livestock_model_prediction(test_image)
                    class_names = list(livestock_cures.keys())
                    predicted_disease = class_names[result_index]
                    
                    # Check if predicted_disease is in livestock_cures
                    if predicted_disease in livestock_cures:
                        cure_link = livestock_cures[predicted_disease]
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
                    else:
                        st.error(f"{translate_text('Prediction', selected_language)} '{predicted_disease}' {translate_text('is not found in the cure dictionary.', selected_language)}")
                else:
                    st.warning(translate_text("Uploaded image isn't a livestock disease or is unclear. Upload a better detailed image of the livestock disease.", selected_language))

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
    ### {translate_text('Livestock Disease Prediction 🐄', selected_language)}  
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
