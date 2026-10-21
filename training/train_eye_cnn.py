from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/eye_dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", default="models/eye_state_mobilenet.keras")
    args = parser.parse_args()

    root = Path(args.data_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            "Expected data/eye_dataset/train and data/eye_dataset/val "
            "with open/closed subdirectories."
        )

    img_size = (96, 96)
    batch = 32

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch,
        label_mode="binary",
        shuffle=True,
        seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch,
        label_mode="binary",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.10),
    ])

    backbone = keras.applications.MobileNetV3Small(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(*img_size, 3))
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v3.preprocess_input(x)
    x = backbone(x, training=False)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved CNN to {output}")


if __name__ == "__main__":
    main()
