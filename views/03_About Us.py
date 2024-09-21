import streamlit as st

st.title("About")
st.markdown("""
            #### About Dataset
            This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
            This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
            A new directory containing 33 test images is created later for prediction purpose.
            #### Content
            1. train (70295 images)
            2. test (33 images)
            3. validation (17572 images)

            """)