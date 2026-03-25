from pathlib import Path
import argparse

import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


BASE_DIR = Path(__file__).resolve().parent
MODEL_JSON_PATH = BASE_DIR / "model.json"
MODEL_WEIGHTS_PATH = BASE_DIR / "model.h5"
TRAIN_DIR = BASE_DIR / "datasets" / "train"


def get_class_labels():
    return sorted(folder.name for folder in TRAIN_DIR.iterdir() if folder.is_dir())


def load_trained_model():
    with open(MODEL_JSON_PATH, "r") as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights(str(MODEL_WEIGHTS_PATH))
    return loaded_model


def predict_image(image_path):
    labels = get_class_labels()
    loaded_model = load_trained_model()

    test_image = image.load_img(str(image_path), target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    prediction = loaded_model.predict(test_image, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))

    print(f"Image: {image_path}")
    print(f"Predicted class: {labels[predicted_index]}")
    print(f"Confidence: {prediction[predicted_index] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Test the Day-10 leaf disease classification model.")
    parser.add_argument("image_path", nargs="?", help="Path to the image to classify")
    args = parser.parse_args()

    image_path = Path(args.image_path) if args.image_path else Path(input("Enter image path: ").strip())

    if not MODEL_JSON_PATH.exists() or not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError("model.json or model.h5 not found in Day-10.")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError("datasets/train folder not found in Day-10.")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    predict_image(image_path)


if __name__ == "__main__":
    main()
