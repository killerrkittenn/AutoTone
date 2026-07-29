# Backend (обучение модели)

Обучает лёгкую свёрточную сеть **с нуля**,
предсказывающую 3 коэффициента коррекции: brightness, contrast,
saturation. Изображения 160×160, компактная архитектура — веса
получаются заметно легче, чем у моделей на базе ImageNet-бэкбонов, что
важно для клиентского бюджета "до 10 МБ".

## Установка

```bash
cd backend
pip install -r requirements.txt
```

## 1. Генерация датасета

Нужна папка с "эталонными" (уже хорошо выглядящими) фотографиями -
скрипт сам портит их случайной яркостью/контрастом/насыщенностью и
запоминает, каким коэффициентом это нужно исправить обратно.

```bash
python generate_dataset.py --source ./reference_images --out ./data --n-per-image 6
```

## 2. Обучение

```bash
python train_model.py --data ./data --epochs 20
```

Результат — `best_model.keras`.

## 3. Проверка качества перед экспортом

```bash
python test_inference.py --model best_model.keras --images ./eval_images --out ./eval_results
```
Сохранит картинки "до/после" рядом для визуальной оценки.

## 4. Экспорт в TF.js для фронтенда

```bash
bash export_model.sh best_model.keras ../frontend/model
```

После этого `frontend/model/` содержит `model.json` + веса, и сайт
автоматически начнёт использовать настоящую модель вместо
эвристического фолбэка.
