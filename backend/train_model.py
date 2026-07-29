import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def load_dataset(data_dir: Path, img_size: int):
    labels_path = data_dir / "labels.csv"
    filenames, targets, groups = [], [], []
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            filenames.append(row["filename"])
            targets.append([float(row["brightness"]), float(row["contrast"]), float(row["saturation"])])
            groups.append(int(row["group"]))

    targets = np.array(targets, dtype=np.float32) / 2.0

    def _load(filename):
        img = tf.io.read_file(tf.strings.join([str(data_dir), "/", filename]))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        return tf.cast(img, tf.float32) / 255.0

    return filenames, targets, np.array(groups), _load


def build_model(img_size: int) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = inputs
    for filters in (16, 32, 64, 96):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(3, activation="sigmoid", name="coefficients")(x)

    model = models.Model(inputs, outputs, name="autotone_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="папка с данными (из generate_dataset.py)")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--out", default="best_model.keras")
    args = parser.parse_args()

    data_dir = Path(args.data)
    filenames, targets, groups, _load = load_dataset(data_dir, args.img_size)

    rng = np.random.RandomState(42)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(len(unique_groups) * args.val_split))
    val_groups = set(unique_groups[:n_val_groups].tolist())

    is_val = np.array([g in val_groups for g in groups])
    train_files = [f for f, v in zip(filenames, is_val) if not v]
    train_targets = targets[~is_val]
    val_files = [f for f, v in zip(filenames, is_val) if v]
    val_targets = targets[is_val]

    print(f"Исходных фото: {len(unique_groups)} (val: {n_val_groups}); "
          f"сэмплов train: {len(train_files)}, val: {len(val_files)}")

    def _make_ds(files, ys, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((files, ys))
        if shuffle:
            ds = ds.shuffle(max(len(files), 1), seed=42)
        ds = ds.map(lambda fn, y: (_load(fn), y), num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.map(lambda img, y: (tf.image.random_flip_left_right(img), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _make_ds(train_files, train_targets, shuffle=True)
    val_ds = _make_ds(val_files, val_targets, shuffle=False)

    model = build_model(args.img_size)
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)
    model.save(args.out)
    print(f"Модель сохранена: {args.out}")


if __name__ == "__main__":
    main()
