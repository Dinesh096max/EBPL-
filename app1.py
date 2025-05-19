import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io
import os # Import the os module

# Title
st.title("EBPL-DS Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) below or upload an image, then hit **Predict**.")

# Check if the model file exists before attempting to load it
model_path = "model.h5"

# Add a print statement to see the current working directory
print(f"Current working directory: {os.getcwd()}")

# Load model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    # Add a print statement to see the path being checked
    print(f"Checking for model file at: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}. Please ensure 'model.h5' is in the same directory as the script.")
        st.stop() # Stop the Streamlit app if the model is not found
    return tf.keras.models.load_model(model_path)

model = load_model(model_path)

# Drawing canvas
canvas_data = st.canvas(
    fill_color="#000000",  # black
    stroke_width=15,
    stroke_color="#FFFFFF",  # white pen on black bg
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Or file uploader
uploaded = st.file_uploader("Or upload a handwritten digit PNG/JPG", type=["png","jpg","jpeg"])

def preprocess(img: Image.Image):
    # convert to grayscale, resize, invert
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    arr = np.array(img)/255.0
    return arr.reshape(1,28,28,1)

# Prediction button
if st.button("Predict"):
    if canvas_data.image_data is not None:
        img = Image.fromarray((canvas_data.image_data[:,:,:3] * 255).astype('uint8'))
    elif uploaded is not None:
        img = Image.open(io.BytesIO(uploaded.read()))
    else:
        st.error("Draw or upload an image first!")
        st.stop()

    x = preprocess(img)
    preds = model.predict(x)
    digit = np.argmax(preds)
    confidence = float(np.max(preds))*100

    st.image(img.resize((140,140)), caption="Input", use_column_width=False)
    st.success(f"Predicted Digit: **{digit}** (confidence: {confidence:.1f}%)")
