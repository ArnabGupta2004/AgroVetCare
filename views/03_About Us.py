import streamlit as st

st.title("About")
st.markdown("""
            #### About Dataset
            This dataset is recreated using offline augmentation from the original dataset.
            This dataset consists of about 90K rgb images of healthy and diseased Crop leaves and Livestock(only cows due to unavailabilty of good quality dataset) which is categorized into 45 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
            #### Content
            """)
tab1, tab2 = st.tabs(["Crops", "Livestocks"])
with tab1:
    st.markdown("""
1. Train Set **(70295 Images)**  
2. Valid Set **(33 Images)**  
3. Validation Set **(17572 Images)**  
    """)
with tab2:
    st.markdown("""
1. Train Set **(567 Images)**  
2. Valid Set **(81 Images)**  
3. Validation Set **(186 Images)**  
    """)
