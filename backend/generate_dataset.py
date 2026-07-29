import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageEnhance

MIN_FACTOR = 0.5
MAX_FACTOR = 1.6


def degrade_image(img: Image.Image):
    """Применяет случайную порчу яркости/контраста/насыщенности.
    Возвращает (испорченное_изображение, {brightness, contrast, saturation})."""
    factors = {
        "brightness": random.uniform(MIN_FACTOR, MAX_FACTOR),
        "contrast": random.uniform(MIN_FACTOR, MAX_FACTOR),
        "saturation": random.uniform(MIN_FACTOR, MAX_FACTOR),
    }

    out = ImageEnhance.Brightness(img).enhance(factors["brightness"])
    out = ImageEnhance.Contrast(out).enhance(factors["contrast"])
    out = ImageEnhance.Color(out).enhance(factors["saturation"])

    # Целевая метка - обратный коэффициент, который "чинит" изображение.
    targets = {k: round(1.0 / v, 4) for k, v in factors.items()}
    return out, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="папка с эталонными изображениями")
    parser.add_argument("--out", default="./data", help="куда сохранить сгенерированный датасет")
    parser.add_argument("--n-per-image", type=int, default=4, help="сколько испорченных вариантов на каждое эталонное фото")
    parser.add_argument("--size", type=int, default=160, help="к какому размеру привести изображения перед сохранением")
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in source_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not image_paths:
        raise SystemExit(f"В {source_dir} не найдено ни одного jpg/png изображения")

    rows = []
    counter = 0
    for source_idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB").resize((args.size, args.size))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path}: {exc}")
            continue

        for _ in range(args.n_per_image):
            degraded, targets = degrade_image(img)
            filename = f"{source_idx:05d}_{counter:06d}.jpg"
            degraded.save(out_dir / filename, quality=90)
            rows.append({"filename": filename, "group": source_idx, **targets})
            counter += 1

        if counter % 200 == 0:
            print(f"...сгенерировано {counter} примеров")

    with open(out_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "group", "brightness", "contrast", "saturation"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Готово: {counter} примеров сохранено в {out_dir}, метки в {out_dir / 'labels.csv'}")


if __name__ == "__main__":
    main()
