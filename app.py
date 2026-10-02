
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="E-Waste Detection",
    page_icon="♻️",
    layout="centered"
)

# ----------------------------
# Load model and class names
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("MobileNetV3_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# ----------------------------
# User interface
# ----------------------------
st.title("♻️ E-Waste Detection System")
st.write(
    "Upload an image of electronic waste to predict its category "
    "using a trained MobileNetV3Large model."
)

st.info(
    "Supported categories: " + ", ".join(class_names)
)

uploaded_file = st.file_uploader(
    "Upload an e-waste image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing image..."):
            # Match the model's training image size
            image_resized = image.resize((224, 224))

            image_array = np.array(image_resized, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)
            probabilities = predictions[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

        st.success("Prediction completed!")

        st.subheader(f"Predicted category: {predicted_class}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.caption(
            "Confidence is the model's predicted score, "
            "not a guarantee that the classification is correct."
        )

        st.subheader("Class-wise scores")

        for index in np.argsort(probabilities)[::-1]:
            st.write(
                f"**{class_names[index]}:** "
                f"{probabilities[index] * 100:.2f}%"
            )
            st.progress(float(probabilities[index]))
