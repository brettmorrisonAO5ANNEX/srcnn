import generate_dataset
import srcnn
import test_model
from pathlib import Path

IMG_SIZE = 256

if __name__ == "__main__":
    # only load dataset if not already
    if not Path("dataset").exists():
        print("creating dataset...")
        generate_dataset.create_dataset(IMG_SIZE, 80, 20)
        print("successfully created dataset")
    else:
        print("dataset already exists, skipping download...")

    # create model
    print("creating model...")
    try:
        model = srcnn.create_model()
    except Exception as e:
        print(f"error creating model: {e}")

    # train model
    print("training model...")
    try:
        # convert dataset to tf.data.Dataset
        ds_train = generate_dataset.get_dataset("dataset/train/LR", "dataset/train/HR", batch_size=8)
        ds_val = generate_dataset.get_dataset("dataset/val/LR", "dataset/val/HR", batch_size=8)

        # train model
        model.fit(ds_train, validation_data=ds_val, epochs=10)
    except Exception as e:
        print(f"error training model: {e}")

    # test model
    test_model.run_test("willy.JPG", model, IMG_SIZE)