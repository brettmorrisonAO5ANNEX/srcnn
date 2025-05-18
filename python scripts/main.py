import generate_dataset
import srcnn
from pathlib import Path

IMG_SIZE = 256

if __name__ == "__main__":
    # only load dataset if not already
    if not Path("dataset").exists():
        print("Creating dataset...")
        generate_dataset.create_dataset(IMG_SIZE, 8, 2)
        print("successfully created dataset")
    else:
        print("Dataset already exists. Skipping download.")

    # create model
    print("Creating model...")
    try:
        model = srcnn.create_model()
    except Exception as e:
        print(f"Error creating model: {e}")

    # train model
    print("Training model...")
    try:
        srcnn.train_model(model, "dataset/train", "dataset/val", epochs=10)
    except Exception as e:
        print(f"Error training model: {e}")

    # test model
    srcnn.test_model()