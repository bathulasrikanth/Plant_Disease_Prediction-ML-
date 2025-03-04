import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model('trained_model.keras')
        image = Image.open(test_image)
        image = image.resize((128, 128))
        input_arr = np.array(image) / 255.0  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)
        prediction = model.predict(input_arr)   
        result_index = np.argmax(prediction)
        return result_index
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About', 'Disease Recognition'])

if app_mode == 'Home':
    st.header('Plant Disease Recognition System')
    image_path = "plant_disease_image.jpg"
    
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
    else:
        st.warning("Image file not found. Displaying a placeholder image.")
        st.image("https://via.placeholder.com/500", use_container_width=True)
    
    st.markdown('''
    ### How It Works
    1. **Upload Image:** Go to **Disease Recognition** and upload an image.
    2. **Analysis:** The system detects potential diseases.
    3. **Results:** Get disease details and recommendations.
    ''')

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    - 87K RGB images of healthy and diseased plant leaves
    - 38 different classes
    - Training: 80%, Validation: 20%, Test: 33 images
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, use_container_width=True)
        
        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                result_index = model_prediction(test_image)
                
                if result_index is not None:
                    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                                  'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                                  'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                                  'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                                  'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                                  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                                  'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                                  'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                                  'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                                  'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                                  'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                                  'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                                  'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                                  'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                                  'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
                    st.success(f"Model Prediction: {class_name[result_index]}")
