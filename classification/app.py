import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
# model = load_model("E:\MACHINE LEARNING\ZUMMIT-AI-ML-LEARNING-PATH\classification\mnist_model.h5")
model = load_model("/workspaces/ZUMMIT-AI-ML-LEARNING-PATH/classification/mnist_model.h5")

# "E:\MACHINE LEARNING\ZUMMIT-AI-ML-LEARNING-PATH\classification\mnist_model.h5"
# Streamlit app title
st.title("MNIST Digit Classification")

# Upload an image
uploaded_file = st.file_uploader("Upload a digit image (28x28 grayscale):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)  # Use LANCZOS for resampling
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    
    # Display results
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(image, caption="Uploaded Image", use_column_width=False, width=400)  # Set image width to 150 pixels

    st.write(f"### Predicted Digit: {predicted_digit}")
