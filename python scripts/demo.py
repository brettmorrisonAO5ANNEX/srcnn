import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import generate_dataset
import srcnn
import test_model

st.set_page_config(
    page_title="Super Resolution Demo",
    layout="wide"
)

# Parameters Sidebar
st.sidebar.title("Training Parameters")
img_size = st.sidebar.number_input("Training Image Size", min_value=64, max_value=512, value=256, step=32)
train_count = st.sidebar.number_input("Num Training Samples", min_value=10, value=80, step=10)
val_count = st.sidebar.number_input("Num Validation Samples", min_value=10, value=20, step=10)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16, 32], index=1)
optimizer = st.sidebar.selectbox("Optimizer", ["adam"])
loss = st.sidebar.selectbox("Loss Function", ["mean_squared_error"])

# Main App
tab1, tab2 = st.tabs(["Output Log", "Model Performance"])
with tab1:
    col1, col2 = st.columns(spec=[0.8, 0.2])
    with col1:
        output_log = st.container(height=300, border=True)
    with col2:
        # Buttons
        create_dataset_button = st.button("Create Dataset")
        train_model_button = st.button("Train Model")
