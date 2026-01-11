import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io
import random

IMG_SIZE = (224, 224)

def ensure_user_dirs(base_dir="user_data"):
    os.makedirs(base_dir, exist_ok=True)

def save_user_image(file_like, label, base_dir="user_data"):
    """
    Save user-labeled image to user_data/<label>/NN.jpg
    file_like: file-like object (BytesIO) or filesystem path
    """
    labdir = os.path.join(base_dir, label)
    os.makedirs(labdir, exist_ok=True)
    idx = len([n for n in os.listdir(labdir) if os.path.isfile(os.path.join(labdir, n))])
    path = os.path.join(labdir, f"{idx+1}.jpg")
    if isinstance(file_like, str):
        img = Image.open(file_like).convert("RGB")
        img.save(path, format="JPEG")
    else:
        img = Image.open(file_like).convert("RGB")
        img.save(path, format="JPEG")
    return path

class Predictor:
    """
    Predictor returns top-K guesses for an image.
    As requested, the predictor will always remain 'a little bad' (funny).
    The `always_bad` flag controls this behavior (the app uses always_bad=True).
    """
    def __init__(self, user_model_path="user_model.h5", classes_path="classes.json"):
        self.user_model_path = user_model_path
        self.classes_path = classes_path
        self.user_model = None
        self.class_names = None
        self._load_user_model()

    def _load_user_model(self):
        if os.path.exists(self.user_model_path) and os.path.exists(self.classes_path):
            try:
                self.user_model = tf.keras.models.load_model(self.user_model_path)
                with open(self.classes_path, "r") as f:
                    self.class_names = json.load(f)
                print("Loaded user model with classes:", self.class_names)
            except Exception as e:
                print("Failed to load user model:", e)
                self.user_model = None
                self.class_names = None
        else:
            self.user_model = None
            self.class_names = None

    def reload_if_updated(self):
        self._load_user_model()

    def _preprocess_pil(self, pil_img):
        img = pil_img.resize(IMG_SIZE)
        arr = np.array(img).astype(np.float32)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def _apply_small_badness(self, labels):
        """
        labels: list[(label:str, score:float)]
        returns modified list with slight, consistent 'funny' errors:
         - sometimes replace top label with a common noun at low confidence
         - slightly lower confidences to sound uncertain
         - occasionally shuffle lower-ranked guesses
        """
        commons = ["banana", "chair", "shoe", "cat", "dog", "tree", "car", "clock", "bottle", "cup", "box"]
        # Replace top label with a random common noun ~35% of the time
        if random.random() < 0.35:
            fake = random.choice(commons)
            labels[0] = (fake, 0.10 + random.random() * 0.08)
            for i in range(1, len(labels)):
                labels[i] = (labels[i][0], max(0.01, labels[i][1] * 0.6))
        else:
            # Keep real labels but lower confidence to be 'hesitant'
            labels = [(lbl, max(0.05, score * 0.5)) for lbl, score in labels]
        # Small chance to shuffle lower-ranked guesses
        if random.random() < 0.25 and len(labels) > 2:
            labels[1], labels[2] = labels[2], labels[1]
        return labels

    def predict_pil(self, pil_img, top=3, always_bad=True):
        """
        Returns list of (label, score).
        If a user model exists, use it; otherwise fall back to MobileNetV2 ImageNet.
        Then (optionally) apply a small persistent badness to keep guesses amusing.
        """
        x = self._preprocess_pil(pil_img)

        # If a user model exists, use it
        if self.user_model is not None and self.class_names:
            preds = self.user_model.predict(x)[0]
            idxs = preds.argsort()[::-1][:top]
            labels = [(self.class_names[i], float(preds[i])) for i in idxs]
            if always_bad:
                labels = self._apply_small_badness(labels)
            return labels

        # Fallback: ImageNet MobileNetV2
        base = MobileNetV2(weights="imagenet")
        preds = base.predict(x)
        decoded = decode_predictions(preds, top=top)[0]
        labels = [(d[1].replace("_", " "), float(d[2])) for d in decoded]
        if always_bad:
            labels = self._apply_small_badness(labels)
        return labels

def fine_tune_on_user_data(user_data_dir, save_model_path="user_model.h5", classes_path="classes.json", epochs=5):
    """
    Build a classifier from user-labeled images in user_data_dir.
    Directory structure: user_data_dir/<label>/*.jpg

    Saves model and a JSON file with ordered class names (index -> label).
    """
    labels = [d for d in os.listdir(user_data_dir) if os.path.isdir(os.path.join(user_data_dir, d))]
    if not labels:
        print("No user data found for fine-tuning.")
        return
    labels.sort()
    num_classes = len(labels)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2,
                                 rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_gen = datagen.flow_from_directory(user_data_dir, target_size=IMG_SIZE, batch_size=8, subset="training")
    val_gen = datagen.flow_from_directory(user_data_dir, target_size=IMG_SIZE, batch_size=8, subset="validation")
    base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = False
    x = base.output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)
    model.save(save_model_path)
    class_indices = train_gen.class_indices
    idx_to_label = {v: k for k, v in class_indices.items()}
    ordered = [idx_to_label[i] for i in range(len(idx_to_label))]
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False)
    print("Saved user model and classes.")
