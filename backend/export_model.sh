#!/usr/bin/env bash
# export_model.sh
# --------------------------------------------------------------------
# Конвертирует обученную Keras-модель (best_model.keras) в TensorFlow.js
# layers-model формат с квантованием весов до float16 - уменьшает
# размер весов примерно вдвое почти без потери точности, что важно
# для бюджета "до 10 МБ суммарного кода" на клиенте.
#
# Использование:
#   bash export_model.sh best_model.keras ../frontend/model
# --------------------------------------------------------------------
set -e

MODEL_PATH="${1:-best_model.keras}"
OUT_DIR="${2:-./tfjs_model}"

pip install --quiet tensorflowjs

tensorflowjs_converter \
  --input_format keras \
  --quantize_float16="*" \
  "$MODEL_PATH" \
  "$OUT_DIR"

echo "Готово: TF.js модель сохранена в $OUT_DIR"
du -sh "$OUT_DIR"
