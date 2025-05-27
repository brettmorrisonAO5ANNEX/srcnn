import requests
from pathlib import Path
import cv2
import random
from tqdm import tqdm
import tensorflow as tf
import shutil

# creates an image dataset of the following structure
# dataset
# | train
#   - HR
#   - LR
# | val
#   - HR
#   - LR
# with an 80/20 split

def create_dataset(size, train_count, val_count, train_progress=None, val_progress=None, log_callback=None):
    dataset_dir = Path("../dataset")
    dataset_dir.mkdir(exist_ok=True)

    # Train dirs
    train_dir = dataset_dir / "train"
    train_hr_dir = train_dir / "HR"
    train_lr_dir = train_dir / "LR"
    train_hr_dir.mkdir(parents=True, exist_ok=True)
    train_lr_dir.mkdir(parents=True, exist_ok=True)

    # Val dirs
    val_dir = dataset_dir / "val"
    val_hr_dir = val_dir / "HR"
    val_lr_dir = val_dir / "LR"
    val_hr_dir.mkdir(parents=True, exist_ok=True)
    val_lr_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://picsum.photos/{size}"

    # Train
    load_subset(train_count, base_url, train_hr_dir, train_lr_dir, size, progress_callback=train_progress, log_callback=log_callback)

    # Vald
    load_subset(val_count, base_url, val_hr_dir, val_lr_dir, size, progress_callback=val_progress, log_callback=log_callback)


def load_subset(count, base_url, hr_dir, lr_dir, size, progress_callback=None, log_callback=None):
    for i in range(count):
        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code == 200:
                file_path = hr_dir / f"{i:04d}_HR.jpg"

                # Save HR image
                with open(file_path, "wb") as f:
                    f.write(response.content)

                # Save LR image
                pixelate(file_path, lr_dir, i, size)

        except Exception as e:
            if log_callback:
                log_callback(f"error downloading image {i+1}: {e}")

        finally:
            if progress_callback:
                progress_callback(i + 1, count)

def pixelate(img_path, lr_dir, index, size):
    hr = cv2.imread(str(img_path))  
    if hr is None:
        print(f"failed to load image: {img_path}")
        return

    # create random pix factor in range [4, 8]
    random_factor = random.randint(4, 8)

    # pixelate HR
    temp = cv2.resize(hr, (size // random_factor, size // random_factor), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(temp, (size, size), interpolation=cv2.INTER_NEAREST)
    file_path = lr_dir / f"{index:04d}_LR.jpg"

    # save LR
    cv2.imwrite(str(file_path), pixelated)
    return

# logic for converting dataset to tf.data.Dataset
def load_image_pair(lr_path, hr_path):
    # Read LR and HR images
    lr = tf.io.read_file(lr_path)
    hr = tf.io.read_file(hr_path)
    lr = tf.image.decode_jpeg(lr, channels=3)
    hr = tf.image.decode_jpeg(hr, channels=3)

    # Convert to float32 and normalize to [0, 1]
    lr = tf.image.convert_image_dtype(lr, tf.float32)
    hr = tf.image.convert_image_dtype(hr, tf.float32)

    return lr, hr

def get_dataset(lr_dir, hr_dir, batch_size=8, shuffle=True):
    lr_paths = sorted([str(p) for p in Path(lr_dir).glob("*.jpg")])
    hr_paths = sorted([str(p) for p in Path(hr_dir).glob("*.jpg")])

    dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths)) # Should be <class 'str'>
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(lr_paths))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def clear_dataset():
    dataset_dir = Path("../dataset")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

