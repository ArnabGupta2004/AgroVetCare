import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import urllib.parse
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image



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

def plant_yes_no(test_image):
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    feature_extractor = vgg16

    svm = joblib.load("leaf_svm_model_jb.joblib")
    
    # Load the image and preprocess it
    img = image.load_img(test_image, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)       
    # Extract features using VGG16
    features = feature_extractor.predict(img_array)
    features = features.reshape(1, -1)
        
    # Predict with OneClassSVM
    prediction = svm.predict(features)
    if prediction[0] == 1:
        return 1
    else:
        return 0
        

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

st.title("Disease Recognition")

dr_ch = option_menu(
    menu_title=None,
    options=["Crop", "LiveStock"],
    icons=["square-fill", "square-fill"],
    default_index=0,
    orientation="horizontal"
)

if dr_ch == "Crop":
    st.header("Crop Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])
    if test_image:
        st.image(test_image, width=200)
        if st.button("Predict"):
            with st.spinner("Please Wait...."):
                yn=1
                #yn=plant_yes_no(test_image)
                if yn==1:
                    result_index, confidence = crop_model_prediction(test_image)
                    class_names = list(crop_cures.keys())
                    predicted_disease = class_names[result_index]
                    #st.write(f"Predicted Disease: {predicted_disease}")
                    
                    # Check if predicted_disease is in crop_cures
                    if predicted_disease in crop_cures:
                        cure_link = crop_cures[predicted_disease]
                        st.success(f"Model is predicting it's a **{predicted_disease}** with **{confidence:.2f}%** confidence.")
                        st.markdown(f"[Find Cure for {predicted_disease}]({cure_link})")
                        
                        # Additional buttons
                        with st.expander("Visit Marketplace"):
                            st.markdown("[Visit Amazon Marketplace](https://www.amazon.in)")

                        with st.expander("Contact Experts"):
                            
                            # Expert 1
                            col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                            with col1:
                                st.markdown("""
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
                                st.markdown("""
                                **Name**: Xyz  
                                **Contact**: [1234567899](tel:1234567899)  
                                **Status**: :red[Offline]  
                                """)
                            with col2:
                                st.image("womanavatar.png", width=50)  # Adjust the width and image path
                    else:
                        st.error(f"Prediction '{predicted_disease}' is not found in the cure dictionary.")
                else:
                    st.warning("Uploaded image isn't a plant/ Upload better detailed image of diseased plant.")

if dr_ch == "LiveStock":
    st.header("Livestock Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])
    if test_image:
        st.image(test_image, width=200)
        if st.button("Predict"):
            with st.spinner("Please Wait...."):
                result_index = livestock_model_prediction(test_image)
                class_names = list(livestock_cures.keys())
                predicted_disease = class_names[result_index]
                st.write(f"Predicted Disease: {predicted_disease}")
                
                # Check if predicted_disease is in livestock_cures
                if predicted_disease in livestock_cures:
                    cure_link = livestock_cures[predicted_disease]
                    st.success(f"Model is predicting it's a **{predicted_disease}**")
                    st.markdown(f"[Find Cure for {predicted_disease}]({cure_link})")
                    
                    # Additional buttons
                    with st.expander("Visit Marketplace"):
                        st.markdown("[Visit Amazon Marketplace](https://www.amazon.in)")

                    with st.expander("Contact Experts"):
                        
                        # Expert 1
                        col1, col2 = st.columns([3, 1])  # 3:1 ratio for left and right columns
                        with col1:
                            st.markdown("""
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
                            st.markdown("""
                            **Name**: Xyz  
                            **Contact**: [1234567899](tel:1234567899)  
                            **Status**: :red[Offline]  
                            """)
                        with col2:
                            st.image("womanavatar.png", width=50)  # Adjust the width and image path

                    
                else:
                    st.error(f"Prediction '{predicted_disease}' is not found in the cure dictionary.")
