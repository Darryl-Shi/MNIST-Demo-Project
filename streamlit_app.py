# streamlit_app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # Output: (batch, 32, 24, 24)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # Output: (batch, 64, 20, 20)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and MaxPool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Output: (batch, 32, 12, 12)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Output: (batch, 64, 4, 4)
        # Flatten
        x = x.view(-1, 64 * 4 * 4)
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
@st.cache_resource
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
