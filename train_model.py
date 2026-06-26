from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(num_classes: int, model_name: str = "mobilenet") -> tf.keras.Model:
    if model_name.lower() == "efficientnet":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.12)(x)
    x = preprocess(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = build_model(num_classes=len(class_names), model_name=args.model_name)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_waste_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save in H5 format because the Streamlit app expects model/waste_model.h5.
    model.save(output_dir / "waste_model.h5")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved model to: {output_dir / 'waste_model.h5'}")
    print(f"Saved class names to: {output_dir / 'class_names.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WasteSort AI image classification model.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset folder with class subfolders.")
    parser.add_argument("--output_dir", type=str, default="model", help="Folder to save waste_model.h5 and class_names.json.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model_name", type=str, default="mobilenet", choices=["mobilenet", "efficientnet"], help="Transfer learning backbone.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
