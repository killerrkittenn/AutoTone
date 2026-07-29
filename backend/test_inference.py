import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance


def predict_coefficients(model, img: Image.Image, img_size: int):
    resized = img.resize((img_size, img_size))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    # Модель отдаёт sigmoid в (0, 1) - см. комментарий в train_model.py -
    # переводим обратно в "естественный" диапазон (0, 2), 1.0 = без изменений.
    brightness, contrast, saturation = model.predict(arr, verbose=0)[0] * 2.0
    return float(brightness), float(contrast), float(saturation)


def apply_coefficients(img: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
    out = ImageEnhance.Brightness(img).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Color(out).enhance(saturation)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True, help="папка с тестовыми изображениями")
    parser.add_argument("--out", default="./eval_results")
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)
    paths = [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    for path in paths:
        img = Image.open(path).convert("RGB")
        b, c, s = predict_coefficients(model, img, args.img_size)
        enhanced = apply_coefficients(img, b, c, s)

        side_by_side = Image.new("RGB", (img.width * 2, img.height))
        side_by_side.paste(img, (0, 0))
        side_by_side.paste(enhanced, (img.width, 0))
        side_by_side.save(out_dir / f"{path.stem}_before_after.jpg", quality=92)

        print(f"{path.name}: brightness={b:.3f} contrast={c:.3f} saturation={s:.3f}")

    print(f"Готово. Сравнения сохранены в {out_dir}")


if __name__ == "__main__":
    main()
