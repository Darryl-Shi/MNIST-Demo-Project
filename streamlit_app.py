# streamlit_app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train_model import Net

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.title("Digit Recognizer")
st.markdown("Draw a digit below (0-9) and the model will predict it.")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Preprocess the image
    img = canvas_result.image_data
    img = Image.fromarray(np.uint8(img)).convert('L')  # Convert RGBA to grayscale
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = np.array(img)
    img = 255 - img  # Invert colors: black background becomes white
    img = (img > 0) * img  # Remove noise

    # Normalize and transform the image
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.1307,), (0.3081,))(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1)

    st.write(f"**Predicted Digit:** {predicted.item()}")
    st.write("**Prediction Probabilities:**")
    prob_data = probabilities.numpy()[0]
    for i, prob in enumerate(prob_data):
        st.write(f"Digit {i}: {prob:.4f}")
    st.bar_chart(prob_data)
