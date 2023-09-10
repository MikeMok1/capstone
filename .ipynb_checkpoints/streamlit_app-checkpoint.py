import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Adjust the model path if needed based on your GitHub repository structure
MODEL_PATH = 'ds_model.h5'
model = load_model(MODEL_PATH)

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess an image."""
    # Convert the PIL image to grayscale
    img = img.convert('L')
    
    # Convert the grayscale PIL image to a numpy array
    img_array = image.img_to_array(img)
    
    # Resize the image
    img_array = image.smart_resize(img_array, target_size)
    
    # Expand dimensions to match model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image if you did so during training
    img_array /= 255.0
    
    return img_array

def get_prediction(img_array):
    """Make predictions on the preprocessed image."""
    predictions = model.predict(img_array)
    return predictions

def main():
    st.title("Chest Xray analysis using neural networks.")
    
    # Upload an image for prediction
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with st.spinner("Making prediction..."):
            # Load and display the image
            image_pil = Image.open(uploaded_image)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image and get predictions
            img_array = preprocess_image(image_pil, target_size=(224, 224))
            predictions = get_prediction(img_array)
            
            # Classify images based on prediction scores
            classifications = ['Pneumonia' if pred[0] > 0.5 else 'Normal' for pred in predictions]
                
            # Display the classifications
            st.write(f"Prediction: {classifications[0]} (Score: {predictions[0][0]:.4f})")

if __name__ == "__main__":
    main()
