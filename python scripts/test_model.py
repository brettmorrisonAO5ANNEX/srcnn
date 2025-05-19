import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

def pixelate_image(img, size, factor=None):
    """Pixelate the input image by downsampling and upsampling using nearest neighbor."""
    if factor is None:
        factor = random.randint(4, 8)
    small_size = size // factor
    temp = cv2.resize(img, (small_size, small_size), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(temp, (size, size), interpolation=cv2.INTER_NEAREST)
    return pixelated

def crop_center(img, size):
    """Crop the center square of given size from the image."""
    h, w = img.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    cropped = img[top:top+size, left:left+size]
    return cropped

def run_test(img_path, model, size):
    # Load and crop image to size x size
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = crop_center(img, size)

    # Convert BGR (cv2 default) to RGB for plotting & model input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create pixelated LR version with random factor
    lr_img = pixelate_image(img_rgb, size)

    # Normalize input for model ([0,1] float32)
    lr_norm = lr_img.astype(np.float32) / 255.0
    lr_norm = np.expand_dims(lr_norm, axis=0)  # batch dimension

    # Run model prediction (super-resolved image)
    sr_img = model.predict(lr_norm)[0]  # remove batch dim

    # Clip and convert output to uint8 for display
    sr_img = np.clip(sr_img, 0, 1)
    sr_img = (sr_img * 255).astype(np.uint8)

    # Plot LR input and SR output side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original (HR)")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Pixelated (LR) Input")
    plt.imshow(lr_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Super-Resolved Output")
    plt.imshow(sr_img)
    plt.axis('off')

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Save figure to results folder with a filename based on input image name
    base_name = os.path.basename(img_path)
    save_path = os.path.join(results_dir, f"result_{base_name}.png")
    plt.savefig(save_path)

    plt.show()
