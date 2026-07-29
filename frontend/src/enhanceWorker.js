/**
 * enhanceWorker.js
 * ------------------------------------------------------------------
 * Работает в отдельном потоке, чтобы UI не блокировался во время
 * инференса модели и обработки изображения.
 *
 * В отличие от простого пайплайна "декодировать -> предсказать ->
 * применить -> отдать Blob", здесь ПОСЛЕ обработки оригинальный
 * ImageBitmap передаётся обратно в основной поток (через transferable
 * objects, без копирования памяти), чтобы пользователь мог на лету
 * подвинуть ползунки и увидеть другой результат без повторного
 * прогона модели - это считает уже основной поток силами
 * OffscreenCanvas (см. photoEnhancer.js -> renderWithAdjustment).
 * ------------------------------------------------------------------
 */

const TFJS_VERSION = '4.20.0';
importScripts(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@${TFJS_VERSION}/dist/tf.min.js`);

const MODEL_INPUT_SIZE = 160; // соответствует размеру, на котором обучалась собственная CNN
const MAX_MEGAPIXELS = 15;

let model = null;
let cancelledTaskId = null;

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      model = await tf.loadLayersModel(msg.modelUrl);
      const warmup = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
      tf.dispose([warmup, model.predict(warmup)]);
    } catch (err) {
      console.warn('[enhanceWorker] модель не найдена, используется эвристический фолбэк:', err.message);
    }
    return;
  }

  if (msg.type === 'cancel') {
    cancelledTaskId = msg.taskId;
    return;
  }

  if (msg.type === 'process') {
    await runPipeline(msg.taskId, msg.buffer, msg.mime);
  }
};

function report(taskId, status, progress) {
  self.postMessage({ type: 'progress', taskId, status, progress });
}

async function runPipeline(taskId, buffer, mime) {
  try {
    report(taskId, 'decoding', 10);
    const bitmap = await createImageBitmap(new Blob([buffer], { type: mime }));

    const megapixels = (bitmap.width * bitmap.height) / 1_000_000;
    if (megapixels > MAX_MEGAPIXELS) {
      bitmap.close();
      throw new Error(`Изображение слишком велико: ${megapixels.toFixed(1)} Мпк (максимум ${MAX_MEGAPIXELS})`);
    }
    if (cancelledTaskId === taskId) return finishCancelled(taskId);

    report(taskId, 'inference', 40);
    const coefficients = await predict(bitmap);
    if (cancelledTaskId === taskId) return finishCancelled(taskId);

    report(taskId, 'enhancing', 70);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.filter = `brightness(${coefficients.brightness * 100}%) contrast(${coefficients.contrast * 100}%) saturate(${coefficients.saturation * 100}%)`;
    ctx.drawImage(bitmap, 0, 0);

    report(taskId, 'encoding', 90);
    const outMime = mime === 'image/png' ? 'image/png' : 'image/jpeg';
    const blob = await canvas.convertToBlob({ type: outMime, quality: 0.92 });

    // Оригинальный bitmap НЕ закрываем - передаём владение им обратно
    // в основной поток (transfer), чтобы включить мгновенную
    // ручную донастройку без повторного инференса.
    self.postMessage({ type: 'result', taskId, blob, coefficients, originalBitmap: bitmap }, [bitmap]);
  } catch (err) {
    self.postMessage({ type: 'error', taskId, error: err.message || String(err) });
  } finally {
    if (cancelledTaskId === taskId) cancelledTaskId = null;
  }
}

function finishCancelled(taskId) {
  cancelledTaskId = null;
  self.postMessage({ type: 'cancelled', taskId });
}

async function predict(bitmap) {
  const canvas = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const imageData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  if (model) {
    return tf.tidy(() => {
      const tensor = tf.browser.fromPixels(imageData).toFloat().div(255).expandDims(0);
      // Модель обучалась на таргетах, поделенных на 2 (см. train_model.py),
      // чтобы они попадали в диапазон обычного sigmoid (0,1) без
      // отдельного слоя масштабирования (Lambda/Rescaling не
      // поддерживаются tensorflowjs_converter). Поэтому здесь нужно
      // умножить сырой выход модели обратно на 2 - без этого шага
      // коэффициенты приходят в 2 раза заниженными (около 0.5 вместо
      // ~1.0 для "без изменений"), из-за чего все фото становятся
      // тусклыми и обесцвеченными.
      const [brightness, contrast, saturation] = model.predict(tensor).dataSync().map((v) => v * 2);
      return { brightness, contrast, saturation };
    });
  }
  return heuristic(imageData);
}

/** Временная заглушка, пока модель не подключена в model/. */
function heuristic(imageData) {
  const { data } = imageData;
  let sumBrightness = 0;
  let sumSaturation = 0;
  const n = data.length / 4;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    sumBrightness += (r + g + b) / 3;
    sumSaturation += max === 0 ? 0 : (max - min) / max;
  }

  const meanBrightness = sumBrightness / n;
  const meanSaturation = sumSaturation / n;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  return {
    brightness: clamp(0.5 / Math.max(meanBrightness, 0.05), 0.7, 1.4),
    contrast: 1.1,
    saturation: clamp(0.45 / Math.max(meanSaturation, 0.05), 0.9, 1.3),
  };
}
