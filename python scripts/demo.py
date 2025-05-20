import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import generate_dataset
import srcnn
import test_model

# UI Sidebar
st.sidebar.title("Training Parameters")
img_size = st.sidebar.number_input("Image Size", min_value=64, max_value=512, value=256, step=32)
train_count = st.sidebar.number_input("Training Samples", min_value=10, value=80, step=10)
val_count = st.sidebar.number_input("Validation Samples", min_value=10, value=20, step=10)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16, 32], index=1)
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd"])
loss = st.sidebar.selectbox("Loss Function", ["mean_squared_error", "mae"])

# Main App
st.title("Super Resolution Model Demo")

if st.button("Create Dataset"):
    if not Path("dataset").exists():
        st.write("Creating dataset...")
        generate_dataset.create_dataset(img_size, train_count, val_count)
        st.success("Dataset created!")
    else:
        st.info("Dataset already exists. Skipping creation.")

if st.button("Train Model"):
    st.write("Creating model...")
    try:
        model = srcnn.create_model(optimizer=optimizer, loss=loss)
    except Exception as e:
        st.error(f"Error creating model: {e}")
        st.stop()

    st.write("Preparing datasets...")
    try:
        ds_train = generate_dataset.get_dataset("dataset/train/LR", "dataset/train/HR", batch_size=batch_size)
        ds_val = generate_dataset.get_dataset("dataset/val/LR", "dataset/val/HR", batch_size=batch_size)
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.stop()

    st.write("Training model...")
    progress = st.progress(0)
    history = None
    try:
        for i in range(epochs):
            history = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=0)
            progress.progress((i + 1) / epochs)
        st.success("Training complete!")
    except Exception as e:
        st.error(f"Error during training: {e}")
        st.stop()

    # Save model to session state
    st.session_state.model = model
    st.session_state.img_size = img_size

#upload and test image
uploaded_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if "model" not in st.session_state:
        st.warning("Please train the model first.")
    else:
        # Read image with OpenCV from uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Display image
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        st.write("Running inference...")
        model = st.session_state.model
        try:
            # Pass OpenCV image directly if supported
            result_img = test_model.run_test(uploaded_file, model, st.session_state.img_size)

            # If result is still OpenCV format (BGR), convert to RGB
            if result_img.shape[2] == 3:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            st.image(result_img, caption="Super-Resolved Output", use_container_width=True)
        except Exception as e:
            st.error(f"Error during testing: {e}")
