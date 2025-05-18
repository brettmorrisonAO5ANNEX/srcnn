import os
import requests
from pathlib import Path
import cv2
import random
from tqdm import tqdm

# creates an image dataset of the following structure
# dataset
# | train
#   - HR
#   - LR
# | val
#   - HR
#   - LR
# with an 80/20 split

def create_dataset(size, train_count, val_count):
    # root
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)

    # test set
    test_dir = dataset_dir / "test"
    test_hr_dir = test_dir / "HR"
    test_lr_dir = test_dir / "LR"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_hr_dir.mkdir(parents=True, exist_ok=True)
    test_lr_dir.mkdir(parents=True, exist_ok=True)

    # validation set
    val_dir = dataset_dir / "val"
    val_hr_dir = val_dir / "HR"
    val_lr_dir = val_dir / "LR"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_hr_dir.mkdir(parents=True, exist_ok=True)
    val_lr_dir.mkdir(parents=True, exist_ok=True)
    
    # url for square image request from picsum
    base_url = f"https://picsum.photos/{size}"

    # load training data
    load_subset(train_count, base_url, Path(test_hr_dir), Path(test_lr_dir), size)
    
    # load validation data
    load_subset(val_count, base_url, Path(val_hr_dir), Path(val_lr_dir), size)
    return

def load_subset(count, base_url, hr_dir, lr_dir, size):
    for i in tqdm(range(count), desc="Downloading images", unit="img"):
        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code == 200:
                file_path = hr_dir / f"{i:04d}_HR.jpg"

                # save HR to dataset/HR
                with open(file_path, "wb") as f:
                    f.write(response.content)

                # save LR to dataset/LR
                pixelate(file_path, lr_dir, i, size)

        except Exception as e:
            tqdm.write(f"Error downloading image {i+1}: {e}")

def pixelate(img_path, lr_dir, index, size):
    hr = cv2.imread(str(img_path))  
    if hr is None:
        print(f"failed to load image: {img_path}")
        return

    # create random pix factor in range [10,16]
    random_factor = random.randint(4, 8)

    # pixelate HR
    temp = cv2.resize(hr, (size // random_factor, size // random_factor), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(temp, (size, size), interpolation=cv2.INTER_NEAREST)
    file_path = lr_dir / f"{index:04d}_LR.jpg"

    # save LR
    cv2.imwrite(str(file_path), pixelated)
    return