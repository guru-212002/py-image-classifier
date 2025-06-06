import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

#loading model from tensorflow for specially classifying images (ie.., weights="imagenet"):
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

# processing image into numpy array of numbers with size of 224 x 224 and expanding dimensions(multiple images):
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

#above processed images are classified and only three predictions are taken:
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decode_prediction = decode_predictions(predictions, top = 3)[0]
        return decode_prediction
        
    except Exception as e:
        st.error(f"Error Classifing Image:{str(e)}")
        return None

    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🤖", layout="centered")
    st.title("AI Image Classifier")
    st.write("Upload a image and AI will classify it contents")
    
    #caching model at load_cached_model:
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()
    
    uploaded_file = st.file_uploader("Choose a image to upload...", type=["jpg","png"])
    
    if uploaded_file is not None:
        
        st.image(
            uploaded_file, caption="Uploaded Image", use_container_width = True
        )
        
        btn = st.button("Classify Image")
        
        if btn:
            with st.spinner("Analyzing Image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)
                
                if predictions:
                    st.subheader("Predictions are:")
                    
                    # predictions has a values like [0(index),"that-object",
                    # 0.9%(value in that presence)]
                    for _, label, score in  predictions:
                        st.write(f"**{label}**: {score:.2%}")
                        
                        
if __name__ == "__main__":
    main()


                    
            
        