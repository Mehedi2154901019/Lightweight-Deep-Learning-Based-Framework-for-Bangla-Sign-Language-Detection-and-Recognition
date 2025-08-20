import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model("new50.keras")

# Define image size (must match what you used for training)
IMG_SIZE = (224, 224)  # Update this to your training image size

# Class labels from 0 to 47 (as per your given indices)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'D', 'Dh', 'NG', 'R', 'T', 'Th', 
               'a', 'aa', 'b', 'bh', 'bisorgo', 'c', 'ch', 'dd', 'ddh', 'e', 'g', 'gh', 'h', 'i', 'j', 
               'jh', 'k', 'kh', 'l', 'm', 'n', 'nng', 'o', 'p', 'ph', 'rr', 's', 'space', 'tt', 'tth', 
               'u', 'y']

# Streamlit interface
st.title("Bangla Sign Language Classifier")
st.write("Upload an image of a Bangla sign language gesture to predict its class.")

# Image upload
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_image, target_size=IMG_SIZE)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Convert the image to array
    img_array = img_to_array(img) / 255.0  # Normalize to the same scale as the training data
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Get the class with highest probability
    predicted_class = class_names[predicted_class_index]  # Map index to class name
    
    # Display prediction
    st.write(f"Predicted Class: {predicted_class}")
