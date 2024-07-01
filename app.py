import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow_hub as hub

# Register custom objects from TensorFlow Hub
custom_objects = {'KerasLayer': hub.KerasLayer}

@st.cache_resource
def load_custom_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("Model loaded successfully!")
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
st.title("Monkeypox, Measles, Chickenpox, and Normal Classification")

st.write("Upload an image and the model will predict the class and accuracy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    predictions = predict_image(image)
    if predictions is not None:
        class_names = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']  
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        predicted_accuracy = predictions[0][predicted_class_index]

        st.write(f'**Predicted class**: {predicted_class} with **accuracy**: {predicted_accuracy:.2%}')
        st.write(f'Class probabilities: {predictions}')
    else:
        st.error("Classification failed. Please ensure the uploaded image is valid and try again.")
