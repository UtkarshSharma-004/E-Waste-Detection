
import json
import os

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


@st.cache_data
def load_reference():
    data = np.load("ewaste_feature_reference.npz")

    return (
        data["centroids"].astype(np.float32),
        data["thresholds"].astype(np.float32),
        data["class_names"].tolist()
    )


# Check required files
required_files = [
    "MobileNetV3_model.keras",
    "class_names.json",
    "ewaste_feature_reference.npz"
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.error(
        "Required files are missing: "
        + ", ".join(missing_files)
    )
    st.info(
        "Place all three files in the same directory as app.py."
    )
    st.stop()


model = load_model()
class_names = load_class_names()
centroids, thresholds, reference_names = load_reference()


# Verify class order
if class_names != reference_names:
    st.error(
        "Class order mismatch between the model and feature "
        "reference. Regenerate ewaste_feature_reference.npz."
    )
    st.stop()


# ----------------------------
# Create feature extraction model
# ----------------------------
try:
    gap_layer = next(
        layer for layer in reversed(model.layers)
        if isinstance(
            layer,
            tf.keras.layers.GlobalAveragePooling2D
        )
    )

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=gap_layer.output
    )

except (StopIteration, ValueError):
    st.error(
        "Could not locate the GlobalAveragePooling2D layer "
        "in the model."
    )
    st.stop()


# ----------------------------
# User interface
# ----------------------------
st.title("♻️ E-Waste Detection System")

st.write(
    "Upload an electronic-waste image to identify its category. "
    "The system first checks whether its learned features match "
    "known e-waste examples."
)

st.info(
    "Supported categories: " + ", ".join(class_names)
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ----------------------------
# Prediction
# ----------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

    except Exception:
        st.error("Unable to open this image.")
        st.stop()

    if st.button("Classify Image", type="primary"):

        with st.spinner("Analyzing image..."):

            # Match the existing model's input size
            image_resized = image.resize((224, 224))

            image_array = np.asarray(
                image_resized,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # --------------------------------
            # Stage 1: Feature-based rejection
            # --------------------------------
            features = feature_model.predict(
                image_array,
                verbose=0
            )[0]

            # Normalize extracted features
            feature_norm = np.linalg.norm(features)

            features = features / max(
                float(feature_norm),
                1e-10
            )

            # Cosine similarity against each class centroid
            similarities = centroids @ features

            candidate_index = int(
                np.argmax(similarities)
            )

            similarity = float(
                similarities[candidate_index]
            )

            threshold = float(
                thresholds[candidate_index]
            )

            # Reject images whose similarity is below
            # the calibrated threshold for the closest class
            rejected = similarity < threshold

            # --------------------------------
            # Stage 2: Original 10-class model
            # --------------------------------
            if not rejected:

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                probabilities = predictions[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = float(
                    probabilities[predicted_index]
                ) * 100


        # --------------------------------
        # Display result
        # --------------------------------
        if rejected:

            st.warning(
                "❌ Not Recognised — Please upload "
                "a valid e-waste image."
            )

            st.write(
                "The image did not sufficiently match "
                "the learned e-waste feature references."
            )

            st.caption(
                "This is an approximate rejection check. "
                "Some genuine e-waste images may be rejected, "
                "and some unrelated images may still be accepted."
            )

        else:

            st.success("Prediction completed!")

            st.subheader(
                f"Predicted category: {predicted_class}"
            )

            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's predicted score, "
                "not a guarantee that the classification is correct."
            )

            st.caption(
                f"Feature similarity: {similarity:.3f} "
                f"(threshold: {threshold:.3f})"
            )

            # Class-wise scores
            st.subheader("Class-wise scores")

            for index in np.argsort(probabilities)[::-1]:

                st.write(
                    f"**{class_names[index]}:** "
                    f"{probabilities[index] * 100:.2f}%"
                )

                st.progress(
                    float(probabilities[index])
                )
