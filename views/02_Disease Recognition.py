import streamlit as st

# Dictionaries for symptoms and corresponding diseases
cattle_symptoms = {
    "Fever": ["Foot and Mouth disease", "Lumpy Skin Disease"],
    "Blisters in the mouth and feet": ["Foot and Mouth disease"],
    "Lumps on the skin": ["Lumpy Skin Disease"],
    "Swollen lymph nodes": ["Lumpy Skin Disease"],
}

poultry_symptoms = {
    "Diarrhea": ["cocci", "salmo"],
    "Dehydration": ["cocci"],
    "Loss of appetite": ["cocci", "salmo"],
    "Neck twisting": ["New Cattle Disease"],
    "Paralysis": ["New Cattle Disease"],
    "Sudden death": ["New Cattle Disease"],
    "Lethargy": ["salmo"],
    "Fever": ["salmo"],
}

goat_symptoms = {
    "Blisters and sores in the mouth": ["Boqueira"],
    "Drooling": ["Boqueira"],
    "Swollen lymph nodes": ["Mal do caroco"],
    "Fever": ["Mal do caroco"],
    "Loss of appetite": ["Mal do caroco"],
}

pig_symptoms = {
    "Fever": [
        "Infected_Bacterial_Erysipelas",
        "Infected_Viral_Foot_and_Mouth_Disease",
    ],
    "Skin lesions": [
        "Infected_Bacterial_Erysipelas",
        "Infected_Bacterial_Greasy_Pig_Disease",
        "Infected_Viral_Swinepox",
    ],
    "Sudden death": ["Infected_Bacterial_Erysipelas"],
    "Oily skin": ["Infected_Bacterial_Greasy_Pig_Disease"],
    "Diarrhea": ["Infected_Bacterial_Greasy_Pig_Disease"],
    "Redness and irritation": ["Infected_Environmental_Dermatitis"],
    "Lesions on the skin": ["Infected_Environmental_Dermatitis"],
    "Red, inflamed skin": ["Infected_Environmental_Sunburn"],
    "Circular, scaly lesions": ["Infected_Fungal_Ringworm"],
    "Scratching and hair loss": ["Infected_Parasitic_Mange"],
    "Blisters on feet and mouth": ["Infected_Viral_Foot_and_Mouth_Disease"],
    "Blisters and pustules": ["Infected_Viral_Swinepox"],
}

bee_symptoms = {
    "Ants around the hive entrance": ["ant_problems"],
    "Stress in bees": ["few_varrao_and_hive_beetles"],
    "Mites on bees": ["few_varrao_and_hive_beetles", "varroa_and_small_hive_beetles"],
    "Increased activity at hive entrance": ["hive_being_robbed"],
    "Dead bees outside": ["hive_being_robbed"],
    "No eggs in the hive": ["missing_queen"],
    "Reduced bee activity": ["missing_queen"],
    "Beetles in the hive": ["varroa_and_small_hive_beetles"],
}

# Animal selection
def get_symptom_dict(animal):
    if animal == "Cattle":
        return cattle_symptoms
    elif animal == "Poultry":
        return poultry_symptoms
    elif animal == "Goat":
        return goat_symptoms
    elif animal == "Pig":
        return pig_symptoms
    elif animal == "Bee":
        return bee_symptoms
    return {}

# Streamlit app
st.title("Livestock Disease Identification System")

# Animal dropdown
animals = ["Cattle", "Poultry", "Goat", "Pig", "Bee"]
selected_animal = st.selectbox("Select the animal:", animals)

# Get symptoms for selected animal
symptom_dict = get_symptom_dict(selected_animal)

if symptom_dict:
    with st.container():
        # Display symptoms as checkboxes
        st.write("### Select Symptoms")
        selected_symptoms = [
            symptom for symptom in symptom_dict if st.checkbox(symptom)
        ]

        if st.button("Predict"):
            if selected_symptoms:
                # Find diseases
                possible_diseases = set()
                for symptom in selected_symptoms:
                    possible_diseases.update(symptom_dict[symptom])

                if possible_diseases:
                    st.subheader("Possible Diseases:")
                    for disease in possible_diseases:
                        st.markdown(f"- {disease}")
                else:
                    st.info("No matching diseases found for the selected symptoms.")
            else:
                st.warning("Please select at least one symptom.")
else:
    st.warning("No symptoms data available for the selected animal.")
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

