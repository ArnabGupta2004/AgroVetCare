import streamlit as st
from googletrans import Translator
from streamlit_option_menu import option_menu

translator = Translator()

# Language selection
col0, colt = st.columns([4, 1], gap="large")
with col0:
    st.header("Welcome to FarmHelp!!")

with colt:
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
    return text  # Return the original text if translation fails


# Page description
st.markdown(translate_text("FarmHelp offers quick insights on disease identification, prevention, and cures to help farmers maintain healthy crops and livestock.", selected_language))

# Option menu
dr_ch = option_menu(
    menu_title=None,
    options=[translate_text("Livestock", selected_language), translate_text("Crop", selected_language)],
    icons=["square-fill", "square-fill"],
    default_index=0,
    orientation="horizontal"
)

# Disease Information for Crops
if dr_ch == translate_text("Crop", selected_language):
    st.subheader(translate_text("Crop Disease Information", selected_language))

    # Disease data structure
    diseases_info = {
        "Apple___Apple_scab": {
            "description": "A fungal disease affecting apple trees, causing scabs on leaves and fruits.",
            "identification": "Dark, scabby spots on leaves and fruit.",
            "precaution": "Plant resistant varieties and ensure proper air circulation.",
            "cure": "Apply fungicides and remove infected leaves."
        },
        "Apple___Black_rot": {
            "description": "A fungal disease causing cankers and fruit rot in apple trees.",
            "identification": "Dark lesions on fruit and leaves, with concentric rings.",
            "precaution": "Prune infected branches and remove diseased fruit.",
            "cure": "Use fungicides and practice good orchard sanitation."
        },
        "Apple___Cedar_apple_rust": {
            "description": "A fungal disease affecting apple and cedar trees, causing orange-yellow spots.",
            "identification": "Yellow-orange spots on the upper side of leaves, with rust on the underside.",
            "precaution": "Remove infected leaves and avoid planting apple trees near cedar trees.",
            "cure": "Apply fungicides in early spring."
        },
        "Cherry_(including_sour)___Powdery_mildew": {
            "description": "A fungal disease causing white, powdery growth on leaves and stems.",
            "identification": "White, powdery spots on leaves, flowers, and young shoots.",
            "precaution": "Prune infected parts and improve air circulation.",
            "cure": "Use fungicides or neem oil."
        },
        "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
            "description": "A fungal disease that causes grayish lesions on corn leaves.",
            "identification": "Small, grayish lesions with yellow halos on leaves.",
            "precaution": "Practice crop rotation and plant resistant varieties.",
            "cure": "Apply fungicides during the growing season."
        },
        "Corn_(maize)___Common_rust_": {
            "description": "A fungal disease that affects corn, causing reddish-brown pustules on leaves.",
            "identification": "Reddish-brown pustules on the upper surface of leaves.",
            "precaution": "Plant resistant varieties and practice crop rotation.",
            "cure": "Use fungicides when symptoms appear."
        },
        "Corn_(maize)___Northern_Leaf_Blight": {
            "description": "A fungal disease causing large, elongated lesions on corn leaves.",
            "identification": "Long, brown lesions on leaves with a characteristic yellow halo.",
            "precaution": "Practice crop rotation and remove infected debris.",
            "cure": "Use resistant varieties and fungicides."
        },
        "Grape___Black_rot": {
            "description": "A fungal disease that causes dark lesions on grapes and leaves.",
            "identification": "Black lesions on leaves, stems, and fruit.",
            "precaution": "Prune infected plants and use disease-free planting material.",
            "cure": "Apply fungicides and remove infected parts."
        },
        "Grape___Esca_(Black_Measles)": {
            "description": "A complex disease affecting grapevines, causing fruit shriveling and dieback.",
            "identification": "Spotted lesions on leaves and shriveled fruit.",
            "precaution": "Prune affected vines and manage vine stress.",
            "cure": "Remove infected wood and apply copper-based fungicides."
        },
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
            "description": "A fungal disease causing small, dark lesions on grapevine leaves.",
            "identification": "Dark, circular spots with yellow halos on leaves.",
            "precaution": "Remove infected leaves and practice good sanitation.",
            "cure": "Use fungicides and improve airflow."
        },
        "Orange___Haunglongbing_(Citrus_greening)": {
            "description": "A bacterial disease affecting citrus, causing yellowing and misshaped fruit.",
            "identification": "Yellowing of leaves and fruit drop.",
            "precaution": "Use certified disease-free plants and control citrus psyllids.",
            "cure": "There is no cure; affected trees should be removed."
        },
        "Peach___Bacterial_spot": {
            "description": "A bacterial disease causing spots and lesions on peach leaves and fruit.",
            "identification": "Watery spots and lesions on leaves and fruit.",
            "precaution": "Prune infected branches and avoid overhead irrigation.",
            "cure": "Apply copper-based bactericides."
        },
        "Pepper,_bell___Bacterial_spot": {
            "description": "A bacterial disease causing dark spots and lesions on pepper leaves.",
            "identification": "Dark lesions surrounded by yellow halos on leaves.",
            "precaution": "Remove infected plant debris and avoid watering leaves.",
            "cure": "Use bactericides and practice crop rotation."
        },
        "Potato___Early_blight": {
            "description": "A fungal disease causing dark spots and lesions on potato leaves.",
            "identification": "Concentric, dark spots on leaves with a yellow halo.",
            "precaution": "Use resistant varieties and avoid overhead irrigation.",
            "cure": "Apply fungicides early in the growing season."
        },
        "Potato___Late_blight": {
            "description": "A devastating fungal disease causing rapid plant decay.",
            "identification": "Water-soaked spots and lesions on leaves, stems, and tubers.",
            "precaution": "Ensure proper spacing and avoid excess moisture.",
            "cure": "Apply fungicides and remove infected plants."
        },
        "Squash___Powdery_mildew": {
            "description": "A fungal disease causing white, powdery growth on squash leaves.",
            "identification": "White, powdery growth on leaves, stems, and flowers.",
            "precaution": "Space plants properly and improve airflow.",
            "cure": "Use fungicides and prune infected parts."
        },
        "Strawberry___Leaf_scorch": {
            "description": "A fungal disease causing brown spots and scorched edges on strawberry leaves.",
            "identification": "Dark brown or black spots with yellow halos.",
            "precaution": "Water plants at the base and ensure good drainage.",
            "cure": "Use fungicides and remove infected leaves."
        },
        "Tomato___Bacterial_spot": {
            "description": "A bacterial disease causing dark, angular spots on tomato leaves.",
            "identification": "Water-soaked lesions with yellow halos on leaves.",
            "precaution": "Avoid overhead irrigation and remove infected plants.",
            "cure": "Apply copper-based bactericides."
        },
        "Tomato___Early_blight": {
            "description": "A fungal disease causing dark, circular spots on tomato leaves.",
            "identification": "Concentric rings around dark lesions on leaves.",
            "precaution": "Practice crop rotation and use resistant varieties.",
            "cure": "Apply fungicides and remove infected leaves."
        },
        "Tomato___Late_blight": {
            "description": "A fungal disease causing rapid decay of tomato plants.",
            "identification": "Water-soaked spots and lesions on leaves and stems.",
            "precaution": "Improve air circulation and practice crop rotation.",
            "cure": "Apply fungicides and remove infected plants."
        },
        "Tomato___Leaf_Mold": {
            "description": "A fungal disease causing mold on tomato leaves.",
            "identification": "Velvety, gray mold on the underside of leaves.",
            "precaution": "Prune plants and avoid overwatering.",
            "cure": "Apply fungicides and improve ventilation."
        },
        "Tomato___Septoria_leaf_spot": {
            "description": "A fungal disease causing small, dark spots on tomato leaves.",
            "identification": "Circular lesions with dark edges on leaves.",
            "precaution": "Remove infected leaves and practice crop rotation.",
            "cure": "Use fungicides and ensure proper plant spacing."
        },
        "Tomato___Spider_mites_Two-spotted_spider_mite": {
            "description": "A pest infestation causing damage to tomato plants.",
            "identification": "Speckled appearance on leaves, with webbing.",
            "precaution": "Use natural predators or miticides.",
            "cure": "Apply miticides or water spray."
        },
        "Tomato___Target_Spot": {
            "description": "A fungal disease causing dark, circular lesions with concentric rings on tomato leaves.",
            "identification": "Circular lesions with dark centers and yellow halos.",
            "precaution": "Space plants properly and remove infected leaves.",
            "cure": "Use fungicides and practice crop rotation."
        },
    }

    # Search for a disease
    search_query = st.text_input(translate_text("Search for a disease", selected_language)).lower()

    # Display diseases based on search or show all
    found_any = False  # Flag to check if any disease matched

    for disease, info in diseases_info.items():
        if search_query in disease.lower() or search_query == "":  # Matching the query against the disease names
            found_any = True
            st.subheader(translate_text(disease, selected_language))
            st.write(translate_text(f"**Description:** {info['description']}", selected_language))
            st.write(translate_text(f"**How to Identify:** {info['identification']}", selected_language))
            st.write(translate_text(f"**Precaution:** {info['precaution']}", selected_language))
            st.write(translate_text(f"**Cure:** {info['cure']}", selected_language))
            st.markdown("---")

    if not found_any:
        st.write(translate_text("No diseases found. Please try a different search term.", selected_language))


# Livestock Disease Information
if dr_ch == translate_text("Livestock", selected_language):
    st.subheader(translate_text("Livestock Disease Information", selected_language))

    # Livestock Diseases Data Structure
    livestock_diseases = {
            "Cattle - Foot and Mouth disease": {
                "description": "A highly contagious viral disease affecting cattle.",
                "identification": "Fever, blisters in the mouth and feet.",
                "precaution": "Vaccinate cattle and avoid contact with infected animals.",
                "cure": "No cure, symptomatic treatment for fever."
            },
            "Cattle - Lumpy Skin Disease": {
                "description": "A viral disease affecting cattle, causing lumps on the skin.",
                "identification": "Lumps on the skin, fever, and swollen lymph nodes.",
                "precaution": "Vaccinate cattle and isolate affected animals.",
                "cure": "No specific cure, symptomatic treatment available."
            },

            "Poultry - cocci": {
                "description": "A parasitic infection caused by coccidia in poultry.",
                "identification": "Diarrhea, dehydration, and loss of appetite.",
                "precaution": "Maintain proper sanitation and provide clean water.",
                "cure": "Use anticoccidial drugs and provide probiotics."
            },

            "Poultry - ncd": {
                "description": "Newcastle Disease is a viral infection in poultry.",
                "identification": "Neck twisting, paralysis, and sudden death.",
                "precaution": "Vaccinate poultry and avoid contact with infected birds.",
                "cure": "No cure, supportive treatment available."
            },
            "Poultry - salmo": {
                "description": "Salmonella infection in poultry.",
                "identification": "Diarrhea, lethargy, and fever.",
                "precaution": "Keep poultry housing clean and avoid overcrowding.",
                "cure": "Use antibiotics for treatment."
            },

            "Goat - Boqueira": {
                "description": "A disease causing mouth ulcers in goats.",
                "identification": "Blisters and sores in the mouth, drooling.",
                "precaution": "Isolate infected animals and disinfect the area.",
                "cure": "Use oral antiseptics and antibiotics."
            },
            "Goat - Mal do caroco": {
                "description": "A parasitic disease causing swelling in the lymph nodes of goats.",
                "identification": "Swollen lymph nodes, fever, and loss of appetite.",
                "precaution": "Vaccinate goats and maintain proper hygiene.",
                "cure": "Use antibiotics and supportive treatment."
            },

            "Pig - Infected_Bacterial_Erysipelas": {
                "description": "A bacterial infection causing high fever and skin lesions.",
                "identification": "Fever, lesions, and sudden death in pigs.",
                "precaution": "Vaccinate pigs and maintain clean pens.",
                "cure": "Use antibiotics such as penicillin."
            },
            "Pig - Infected_Bacterial_Greasy_Pig_Disease": {
                "description": "A bacterial infection causing skin lesions and greasiness in pigs.",
                "identification": "Oily skin, lesions, and diarrhea.",
                "precaution": "Provide proper nutrition and clean housing.",
                "cure": "Use antibiotics and topical treatments."
            },
            "Pig - Infected_Environmental_Dermatitis": {
                "description": "Skin inflammation caused by poor environmental conditions.",
                "identification": "Redness, irritation, and lesions on the skin.",
                "precaution": "Improve living conditions and hygiene.",
                "cure": "Use topical ointments and maintain clean environments."
            },
            "Pig - Infected_Environmental_Sunburn": {
                "description": "Skin damage caused by overexposure to the sun.",
                "identification": "Red, inflamed skin, especially on ears and back.",
                "precaution": "Provide shade and avoid prolonged sun exposure.",
                "cure": "Use soothing ointments and ensure shade."
            },
            "Pig - Infected_Fungal_Pityriasis_Rosea": {
                "description": "A fungal infection causing skin lesions.",
                "identification": "Red, circular patches on the skin.",
                "precaution": "Maintain proper hygiene and avoid overcrowding.",
                "cure": "Use antifungal treatments."
            },
            "Pig - Infected_Fungal_Ringworm": {
                "description": "A fungal infection affecting pigs' skin.",
                "identification": "Circular, scaly lesions on the skin.",
                "precaution": "Isolate affected pigs and disinfect the area.",
                "cure": "Apply antifungal medications."
            },
            "Pig - Infected_Parasitic_Mange": {
                "description": "A parasitic infection causing intense itching and hair loss.",
                "identification": "Scratching, hair loss, and skin lesions.",
                "precaution": "Use mite control treatments and disinfect the area.",
                "cure": "Use acaricides and topical treatments."
            },
            "Pig - Infected_Viral_Foot_and_Mouth_Disease": {
                "description": "A viral infection causing fever and blisters on the feet and mouth.",
                "identification": "Fever, blisters, and lameness.",
                "precaution": "Vaccinate pigs and isolate infected animals.",
                "cure": "No cure, symptomatic treatment available."
            },
            "Pig - Infected_Viral_Swinepox": {
                "description": "A viral infection causing skin lesions.",
                "identification": "Blisters and pustules on the skin.",
                "precaution": "Vaccinate pigs and maintain good hygiene.",
                "cure": "No specific cure, supportive care recommended."
            },

            "Bee - ant_problems": {
                "description": "Ants invading the hive, disturbing the bees.",
                "identification": "Ants around the hive entrance and inside the hive.",
                "precaution": "Use ant traps and maintain hive cleanliness.",
                "cure": "Remove ants manually and use natural repellents."
            },
            "Bee - few_varrao_and_hive_beetles": {
                "description": "A combination of varroa mites and hive beetles affecting bees.",
                "identification": "Bees exhibiting stress, mites on the bees.",
                "precaution": "Use chemical treatments and maintain hive hygiene.",
                "cure": "Treat with miticides and beetle traps."
            },
            "Bee - hive_being_robbed": {
                "description": "Other bees robbing honey from a weak hive.",
                "identification": "Increased activity at the hive entrance, dead bees outside.",
                "precaution": "Increase hive strength and remove weak colonies.",
                "cure": "Secure the hive and prevent entry of robbing bees."
            },
            "Bee - missing_queen": {
                "description": "A situation where the bee queen is absent or dead.",
                "identification": "No eggs in the hive and reduced bee activity.",
                "precaution": "Monitor the queen's health and replace her if needed.",
                "cure": "Introduce a new queen to the hive."
            },
            "Bee - varroa_and_small_hive_beetles": {
                "description": "Infection caused by varroa mites and small hive beetles.",
                "identification": "Mites on bees and beetles in the hive.",
                "precaution": "Use mite control measures and beetle traps.",
                "cure": "Use miticides and beetle traps."
            }
    }

    cattle_diseases = {
        "Foot and Mouth disease": {
            "description": "A highly contagious viral disease affecting cattle.",
            "identification": "Fever, blisters in the mouth and feet.",
            "precaution": "Vaccinate cattle and avoid contact with infected animals.",
            "cure": "No cure, symptomatic treatment for fever."
        },
        "Lumpy Skin Disease": {
            "description": "A viral disease affecting cattle, causing lumps on the skin.",
            "identification": "Lumps on the skin, fever, and swollen lymph nodes.",
            "precaution": "Vaccinate cattle and isolate affected animals.",
            "cure": "No specific cure, symptomatic treatment available."
        }
    }

    poultry_diseases = {
        "cocci": {
            "description": "A parasitic infection caused by coccidia in poultry.",
            "identification": "Diarrhea, dehydration, and loss of appetite.",
            "precaution": "Maintain proper sanitation and provide clean water.",
            "cure": "Use anticoccidial drugs and provide probiotics."
        },
        "ncd": {
            "description": "Newcastle Disease is a viral infection in poultry.",
            "identification": "Neck twisting, paralysis, and sudden death.",
            "precaution": "Vaccinate poultry and avoid contact with infected birds.",
            "cure": "No cure, supportive treatment available."
        },
        "salmo": {
            "description": "Salmonella infection in poultry.",
            "identification": "Diarrhea, lethargy, and fever.",
            "precaution": "Keep poultry housing clean and avoid overcrowding.",
            "cure": "Use antibiotics for treatment."
        }
    }

    pig_diseases = {
        "Infected_Bacterial_Erysipelas": {
            "description": "A bacterial infection causing high fever and skin lesions.",
            "identification": "Fever, lesions, and sudden death in pigs.",
            "precaution": "Vaccinate pigs and maintain clean pens.",
            "cure": "Use antibiotics such as penicillin."
        },
        "Infected_Bacterial_Greasy_Pig_Disease": {
            "description": "A bacterial infection causing skin lesions and greasiness in pigs.",
            "identification": "Oily skin, lesions, and diarrhea.",
            "precaution": "Provide proper nutrition and clean housing.",
            "cure": "Use antibiotics and topical treatments."
        },
        "Infected_Environmental_Dermatitis": {
            "description": "Skin inflammation caused by poor environmental conditions.",
            "identification": "Redness, irritation, and lesions on the skin.",
            "precaution": "Improve living conditions and hygiene.",
            "cure": "Use topical ointments and maintain clean environments."
        },
        "Infected_Environmental_Sunburn": {
            "description": "Skin damage caused by overexposure to the sun.",
            "identification": "Red, inflamed skin, especially on ears and back.",
            "precaution": "Provide shade and avoid prolonged sun exposure.",
            "cure": "Use soothing ointments and ensure shade."
        },
        "Infected_Fungal_Pityriasis_Rosea": {
            "description": "A fungal infection causing skin lesions.",
            "identification": "Red, circular patches on the skin.",
            "precaution": "Maintain proper hygiene and avoid overcrowding.",
            "cure": "Use antifungal treatments."
        },
        "Infected_Fungal_Ringworm": {
            "description": "A fungal infection affecting pigs' skin.",
            "identification": "Circular, scaly lesions on the skin.",
            "precaution": "Isolate affected pigs and disinfect the area.",
            "cure": "Apply antifungal medications."
        },
        "Infected_Parasitic_Mange": {
            "description": "A parasitic infection causing intense itching and hair loss.",
            "identification": "Scratching, hair loss, and skin lesions.",
            "precaution": "Use mite control treatments and disinfect the area.",
            "cure": "Use acaricides and topical treatments."
        },
        "Infected_Viral_Foot_and_Mouth_Disease": {
            "description": "A viral infection causing fever and blisters on the feet and mouth.",
            "identification": "Fever, blisters, and lameness.",
            "precaution": "Vaccinate pigs and isolate infected animals.",
            "cure": "No cure, symptomatic treatment available."
        },
        "Infected_Viral_Swinepox": {
            "description": "A viral infection causing skin lesions.",
            "identification": "Blisters and pustules on the skin.",
            "precaution": "Vaccinate pigs and maintain good hygiene.",
            "cure": "No specific cure, supportive care recommended."
        }
    }

    goat_diseases = {
        "Boqueira": {
            "description": "A disease causing mouth ulcers in goats.",
            "identification": "Blisters and sores in the mouth, drooling.",
            "precaution": "Isolate infected animals and disinfect the area.",
            "cure": "Use oral antiseptics and antibiotics."
        },
        "Mal do caroco": {
            "description": "A parasitic disease causing swelling in the lymph nodes of goats.",
            "identification": "Swollen lymph nodes, fever, and loss of appetite.",
            "precaution": "Vaccinate goats and maintain proper hygiene.",
            "cure": "Use antibiotics and supportive treatment."
        }
    }

    bee_diseases = {
        "ant_problems": {
            "description": "Ants invading the hive, disturbing the bees.",
            "identification": "Ants around the hive entrance and inside the hive.",
            "precaution": "Use ant traps and maintain hive cleanliness.",
            "cure": "Remove ants manually and use natural repellents."
        },
        "few_varrao_and_hive_beetles": {
            "description": "A combination of varroa mites and hive beetles affecting bees.",
            "identification": "Bees exhibiting stress, mites on the bees.",
            "precaution": "Use chemical treatments and maintain hive hygiene.",
            "cure": "Treat with miticides and beetle traps."
        },
        "hive_being_robbed": {
            "description": "Other bees robbing honey from a weak hive.",
            "identification": "Increased activity at the hive entrance, dead bees outside.",
            "precaution": "Increase hive strength and remove weak colonies.",
            "cure": "Secure the hive and prevent entry of robbing bees."
        },
        "missing_queen": {
            "description": "A situation where the bee queen is absent or dead.",
            "identification": "No eggs in the hive and reduced bee activity.",
            "precaution": "Monitor the queen's health and replace her if needed.",
            "cure": "Introduce a new queen to the hive."
        },
        "varroa_and_small_hive_beetles": {
            "description": "Infection caused by varroa mites and small hive beetles.",
            "identification": "Mites on bees and beetles in the hive.",
            "precaution": "Use mite control measures and beetle traps.",
            "cure": "Use miticides and beetle traps."
        }
    }

    # Livestock Disease Selection
    livestock_type =st.selectbox(
        translate_text("Select Livestock Category:", selected_language),
        [
            translate_text("All", selected_language),
            translate_text("Cattle", selected_language),
            translate_text("Poultry", selected_language),
            translate_text("Pig", selected_language),
            translate_text("Goat", selected_language),
            translate_text("Bees", selected_language)
        ],
        index=0  # Set Cattle as the default option
    )
    
    if livestock_type == translate_text("All", selected_language):
        disease_info = livestock_diseases

    elif livestock_type == translate_text("Cattle", selected_language):
        disease_info = cattle_diseases

    elif livestock_type == translate_text("Poultry", selected_language):
        disease_info = poultry_diseases

    elif livestock_type == translate_text("Pig", selected_language):
        disease_info = pig_diseases

    elif livestock_type == translate_text("Goat", selected_language):
        disease_info = goat_diseases

    elif livestock_type == translate_text("Bees", selected_language):
        disease_info = bee_diseases



    search_query = st.text_input(translate_text("Search for a disease", selected_language)).lower()

    # Display diseases based on search or show all
    found_any = False  # Flag to check if any disease matched

    for disease, info in disease_info.items():
        if search_query in disease.lower() or search_query == "":  # Matching the query against the disease names
            found_any = True
            st.subheader(translate_text(disease, selected_language))
            st.write(translate_text(f"**Description:** {info['description']}", selected_language))
            st.write(translate_text(f"**How to Identify:** {info['identification']}", selected_language))
            st.write(translate_text(f"**Precaution:** {info['precaution']}", selected_language))
            st.write(translate_text(f"**Cure:** {info['cure']}", selected_language))
            st.markdown("---")

    if not found_any:
        st.write(translate_text("No diseases found. Please try a different search term.", selected_language))
