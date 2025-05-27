import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import generate_dataset
import srcnn
import test_model
import custom_loss
import matplotlib.pyplot as plt

# initialize session state variables
if "dataset_ready" not in st.session_state:
    st.session_state.dataset_ready = False
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "training_finished" not in st.session_state:
    st.session_state.training_finished = False

# load css styliing
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("../styles/streamlit.css")

# Parameters Sidebar
st.sidebar.title("Training Parameters")
img_size = st.sidebar.number_input("Training Image Size", min_value=64, max_value=512, value=256, step=32)
train_count = st.sidebar.number_input("Num Training Samples", min_value=10, value=80, step=10)
val_count = st.sidebar.number_input("Num Validation Samples", min_value=10, value=20, step=10)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16, 32], index=1)
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "rmsprop", "sgd"], index=0)
loss = st.sidebar.selectbox("Loss Function", ["mean_squared_error", "mean_absolute_error", 
                                               "huber_loss", "ssim"], index=0)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Output Log", "Model Performance", "Test Output", "Preview Dataset"])
with tab1:
    output_log = st.container(height=600, border=True)

with tab2:
    # Check that all required metrics are available
    has_loss = "train_loss" in st.session_state and "val_loss" in st.session_state
    has_acc = "train_acc" in st.session_state and "val_acc" in st.session_state

    if has_loss or has_acc:
        col1, col2 = st.columns(2)

        # Plot Loss in left column
        if has_loss:
            with col1:
                train_loss = st.session_state.train_loss
                val_loss = st.session_state.val_loss
                epochs = range(1, len(train_loss) + 1)

                st.header("Loss vs Epoch")
                fig_loss, ax1 = plt.subplots()
                ax1.plot(epochs, train_loss, label='Train Loss', marker='o')
                ax1.plot(epochs, val_loss, label='Validation Loss', marker='o')
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.set_xticks(epochs)
                ax1.legend()
                st.pyplot(fig_loss)

        # Plot Accuracy in right column
        if has_acc:
            with col2:
                train_acc = st.session_state.train_acc
                val_acc = st.session_state.val_acc
                if any(v is not None for v in train_acc) and any(v is not None for v in val_acc):
                    epochs = range(1, len(train_acc) + 1)

                    st.header("Accuracy vs Epoch")
                    fig_acc, ax2 = plt.subplots()
                    ax2.plot(epochs, train_acc, label='Train Accuracy', marker='o')
                    ax2.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy")
                    ax2.set_xticks(epochs)
                    ax2.legend()
                    st.pyplot(fig_acc)

    else:
        st.warning("No performance data available, train model first")

with tab3:
    if "hr_img" in st.session_state and "lr_img" in st.session_state and "sr_img" in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(st.session_state.hr_img, caption="High Resolution (HR)", use_container_width=True)

        with col2:
            st.image(st.session_state.lr_img, caption="Low Resolution (LR)", use_container_width=True)

        with col3:
            st.image(st.session_state.sr_img, caption="Super-Resolved (SR)", use_container_width=True)
    else:
        st.warning("No test results available, please run 'Test Model' first")
            
with tab4:
    # Dropdowns for dataset and resolution
    dataset_type = st.selectbox("Select Dataset", ["train", "val"])
    resolution_type = st.selectbox("Select Resolution", ["HR", "LR"])

    folder_path = Path("../dataset") / dataset_type / resolution_type

    if not folder_path.exists():
        st.warning("No dataset has been generated, please generate the dataset before previewing")
    else:
        # List image files
        image_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

        if not image_files:
            st.warning("No images found in the selected folder")
        else:
            # Slider for number of images per page
            images_per_page = st.slider("Images per page", min_value=4, max_value=40, value=12, step=4)

            # Slider for page number
            max_page = (len(image_files) - 1) // images_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=max_page, value=1)

            # Compute range
            start_idx = (page - 1) * images_per_page
            end_idx = start_idx + images_per_page
            files_to_display = image_files[start_idx:end_idx]

            # Display images in a 4-column grid
            cols = st.columns(4)
            for i, image_path in enumerate(files_to_display):
                # Read and convert BGR to RGB
                img = cv2.imread(str(image_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with cols[i % 4]:
                    st.image(img_rgb, caption=image_path.name, use_container_width=True)

# Control buttons
st.sidebar.markdown("---")
if st.sidebar.button("Refresh Session"):
    with output_log: 
        generate_dataset.clear_dataset()
        st.session_state.clear() 
        st.success("Session refreshed. All data cleared.")

if st.sidebar.button("Generate Dataset"):
    with output_log:
        if not st.session_state.dataset_ready:
            st.write("\>> Creating dataset")

            # Initialize progress bars
            train_prog = st.progress(0, text=f"Downloading training set: 0%")
            val_prog = st.progress(0, text=f"Downloading validation set: 0%")

            # Progress update functions
            def update_train_progress(current, total):
                train_prog.progress(current / total, text=f"Downloading training set: {int((current / total)*100)}%")

            def update_val_progress(current, total):
                val_prog.progress(current / total, text=f"Downloading validation set: {int((current / total)*100)}%")

            def log_message(message):
                st.write(message)

            try:
                generate_dataset.create_dataset(
                    img_size, train_count, val_count,
                    train_progress=update_train_progress,
                    val_progress=update_val_progress,
                    log_callback=log_message
                )

                st.session_state.dataset_ready = True
                st.success("Dataset ready!.")

            except Exception as e:
                st.error(f"Error generating dataset: {e}")
        else:
            st.info("Dataset already generated. Skipping.")

if st.sidebar.button("Train Model"):
    with output_log:
        if st.session_state.dataset_ready:
            st.write("\>> Creating model")
            try:
                model = srcnn.create_model(optimizer=optimizer, loss=custom_loss.ssim if loss == "ssim" else loss)
                st.session_state.model = model
                st.session_state.model_ready = True
            except Exception as e:
                st.error(f"Error creating model: {e}")
                st.stop()

            st.write("\>> Preparing datasets")
            try:
                ds_train = generate_dataset.get_dataset("../dataset/train/LR", "../dataset/train/HR", batch_size=batch_size)
                ds_val = generate_dataset.get_dataset("../dataset/val/LR", "../dataset/val/HR", batch_size=batch_size)
                st.session_state.ds_train = ds_train
                st.session_state.ds_val = ds_val
                st.session_state.img_size = img_size
            except Exception as e:
                st.error(f"Error loading datasets: {e}")
                st.stop()

            st.write("\>> Training model")
            try:
                train_loss_history = []
                val_loss_history = []
                train_acc_history = []
                val_acc_history = []

                for i in range(epochs):
                    history = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=0)

                    train_loss = history.history.get("loss", [None])[0]
                    val_loss = history.history.get("val_loss", [None])[0]
                    train_acc = history.history.get("accuracy", [None])[0]
                    val_acc = history.history.get("val_accuracy", [None])[0]

                    train_loss_history.append(train_loss)
                    val_loss_history.append(val_loss)
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)

                    with st.container(border=True):
                        st.write(f"Epoch {i+1}/{epochs} complete")
                        if train_acc is not None and val_acc is not None:
                            st.write(f"training accuracy: {train_acc:.4f}")
                            st.write(f"validation accuracy: {val_acc:.4f}")
                        else:
                            st.write("accuracy metrics not available in history")

                st.session_state.train_loss = train_loss_history
                st.session_state.val_loss = val_loss_history
                st.session_state.train_acc = train_acc_history
                st.session_state.val_acc = val_acc_history
                st.session_state.training_finished = True

                st.success("Training complete!")

            except Exception as e:
                st.error(f"Error during training: {e}")
                st.stop()

        else:
            st.warning("Please generate dataset first")

if st.sidebar.button("Test Model"):
    with output_log:
        if st.session_state.training_finished:
            try:
                hr_test, lr_test, sr_test = test_model.run_test(
                    "../willy.JPG", 
                    st.session_state.model, 
                    st.session_state.img_size
                )

                st.session_state.hr_img = hr_test
                st.session_state.lr_img = lr_test
                st.session_state.sr_img = sr_test

                st.session_state.test_ready = True
                st.success("Model testing complete!")

            except Exception as e:
                st.error(f"Error during model test: {e}")
        else:
            st.warning("Please train model first")
