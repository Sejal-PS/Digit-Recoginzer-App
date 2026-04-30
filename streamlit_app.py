"""
Streamlit App for Digit Recognition
This app uses a PyTorch CNN model to recognize handwritten digits (0-9).
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io

# Page configuration
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

# Define the CNN model (same architecture as training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    """Load the trained PyTorch model"""
    model = CNN()
    try:
        model.load_state_dict(torch.load('digit_recognition_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'digit_recognition_model.pth' not found. Please ensure the model is trained and available.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Invert colors if needed (white background -> black)
    image = ImageOps.invert(image)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Add channel dimension and batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=0)
    
    return torch.tensor(img_array)

def predict(model, image):
    """Make prediction on the image"""
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_digit].item()
    
    return predicted_digit, confidence, probabilities[0].numpy()

# Main app
def main():
    st.title("🔢 Handwritten Digit Recognizer")
    st.markdown("Upload an image of a handwritten digit (0-9) and the AI will recognize it!")
    
    # Load model
    model = load_model()
    
    if model is not None:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Digit', use_container_width=True)
            
            # Make prediction
            if st.button("Recognize Digit", type="primary"):
                with st.spinner("Recognizing..."):
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    predicted_digit, confidence, all_probs = predict(model, processed_image)
                
                with col2:
                    st.subheader("Prediction Result")
                    
                    # Display the predicted digit prominently
                    st.markdown(f"<h1 style='text-align: center; font-size: 72px; color: #4CAF50;'>{predicted_digit}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>Confidence: {confidence*100:.2f}%</p>", unsafe_allow_html=True)
                
                # Show probability distribution
                st.subheader("Probability Distribution")
                probs_df = {f"Digit {i}": f"{p*100:.1f}%" for i, p in enumerate(all_probs)}
                st.bar_chart(all_probs, horizontal=True)
                
                # Show detailed probabilities
                with st.expander("See all probabilities"):
                    for i, prob in enumerate(all_probs):
                        st.progress(prob)
                        st.text(f"Digit {i}: {prob*100:.2f}%")
        
        # Demo section
        st.divider()
        st.subheader("📷 Or try with camera")
        
        if st.button("Open Camera"):
            st.info("Use the file uploader above to upload a photo from your camera.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Powered by PyTorch CNN Model | Trained on MNIST Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()