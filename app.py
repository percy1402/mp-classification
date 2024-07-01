import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow_hub as hub

# Set Streamlit to use dark theme
st.set_page_config(layout="centered", page_title="Diag-Assist", page_icon="🩺")

# CSS styles to center title, justify text, and increase font size
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        text-transform: uppercase;
        margin-bottom: 0.5em;
    }
    .paragraph {
        text-align: justify;
    }
    .spaced-paragraph {
        margin-bottom: 1.5em;
    }
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    .diagnosis {
        font-size: 1.5em;
        font-weight: bold;
        text-align: justify;
    }
    .disclaimer {
        font-weight: bold;
        color: red;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

# Register custom objects from TensorFlow Hub
custom_objects = {'KerasLayer': hub.KerasLayer}

@st.cache_resource
def load_custom_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("The application is ready to use!")
        return model
    except Exception as e:
        st.error("There was an error loading the model. Please check the model path and ensure it is a valid TensorFlow model.")
        st.error(f"Technical details: {e}")
        return None

# Load the pre-trained model (ensure this is only done once)
model = load_custom_model('final-bit.h5')

# Function to preprocess the uploaded image
def preprocess_image(img, target_size=(224, 224)):
    try:
        img = img.resize(target_size)
        # Ensure the image has 3 channels (RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize to [0,1]
        return img_array
    except Exception as e:
        st.error("Error during image preprocessing. Please ensure the uploaded file is a valid image.")
        st.error(f"Technical details: {e}")
        return None

# Function to predict the presence of monkeypox
def predict_image(img):
    img_array = preprocess_image(img)
    if img_array is not None:
        try:
            predictions = model.predict(img_array)
            return predictions
        except Exception as e:
            st.error("Error during prediction. Please try again with a different image or check the model.")
            st.error(f"Technical details: {e}")
            return None
    else:
        return None

# Streamlit App
st.markdown("<h1 class='title'>Diag-Assist</h1>", unsafe_allow_html=True)
st.markdown("""
<p class="paragraph spaced-paragraph">An AI-powered diagnostic tool for identifying Monkeypox, Measles, Chickenpox, and normal skin conditions.</p>
<p class="paragraph">Upload a clear picture of the patient's affected skin area. The model will analyze the image and provide a diagnosis along with the confidence level. Please note that this tool is for research purposes only and should not replace professional medical advice.</p>
""", unsafe_allow_html=True)
st.markdown("  ", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload the picture of the patient's affected area...", type=["jpg", "jpeg", "png"], help="Drag and drop the image here or click to browse.")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='The affected area image', use_column_width=False, width=300, output_format="PNG")
    
    predictions = predict_image(image)
    if predictions is not None:
        class_names = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']  
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        predicted_accuracy = predictions[0][predicted_class_index]

        st.markdown(f"<p class='diagnosis'>The diagnosis is {predicted_class.upper()} with accuracy {predicted_accuracy:.2%}.</p>", unsafe_allow_html=True)
    else:
        st.error("Classification failed. Please ensure the uploaded image is valid and try again.")

st.markdown("""
---
<p class="disclaimer">Disclaimer :</p> This application is a research project and has not been verified by any medical organization. It is not intended to replace professional medical advice, diagnosis, or treatment. The results provided by this tool should be used for informational purposes only and should be discussed with a qualified healthcare professional for medical advice.
""", unsafe_allow_html=True)
