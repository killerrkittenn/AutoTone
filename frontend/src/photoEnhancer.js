/**
 * photoEnhancer.js
 * ------------------------------------------------------------------
 * Публичный API модуля. Реализует обязательный контракт из ТЗ проекта
 * (метод постановки задачи / статуса / прерывания / получения
 * результата + событие изменения статуса), плюс дополнительную
 * возможность, которой не было в первой версии фронтенда:
 * интерактивную ручную донастройку коэффициентов после того, как
 * модель предложила свои значения - без повторного обращения к
 * модели, пересчёт идёт мгновенно на клиенте через canvas-фильтр.
 *
 * Методы контракта ТЗ:
 *   - submitTask(file)        -> Promise<taskId>
 *   - getTaskStatus(taskId)   -> {status, progress}
 *   - cancelTask(taskId)      -> Promise<{success}>
 *   - getResult(taskId)       -> Promise<Blob>
 *
 * Событие:
 *   - 'statuschange' -> CustomEvent<{taskId, status, progress}>
 *
 * Дополнительно (не из ТЗ, авторская доработка):
 *   - getSuggestedCoefficients(taskId) -> {brightness, contrast, saturation}
 *   - renderWithAdjustment(taskId, {brightness, contrast, saturation}) -> Promise<Blob>
 * ------------------------------------------------------------------
 */

const SUPPORTED_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.heic', '.heif'];
const MAX_MEGAPIXELS = 15;
const MAX_PROCESSING_MS = 30_000;

export class PhotoEnhancer extends EventTarget {
  #worker;
  #tasks = new Map();
  #queue = [];
  #activeTaskId = null;
  #watchdog = null;

  constructor({ workerUrl = new URL('./enhanceWorker.js', import.meta.url), modelUrl = '../model/model.json' } = {}) {
    super();
    this.#worker = new Worker(workerUrl);
    this.#worker.addEventListener('message', (e) => this.#handleWorkerMessage(e.data));
    this.#worker.postMessage({ type: 'init', modelUrl });
  }

  async submitTask(file) {
    if (!(file instanceof Blob)) throw new TypeError('Ожидается File/Blob с изображением');
    if (!this.#hasSupportedExtension(file)) {
      throw new Error('Неподдерживаемый формат. Разрешены: JPG, PNG, HEIC, BMP');
    }

    const taskId = crypto.randomUUID();
    this.#tasks.set(taskId, {
      status: 'queued',
      progress: 0,
      file,
      resultBlob: null,
      coefficients: null,
      originalBitmap: null,
      error: null,
    });
    this.#queue.push(taskId);
    this.#emit(taskId);
    this.#advanceQueue();
    return taskId;
  }

  getTaskStatus(taskId) {
    const t = this.#requireTask(taskId);
    return { status: t.status, progress: t.progress };
  }

  async cancelTask(taskId) {
    const t = this.#tasks.get(taskId);
    if (!t || ['done', 'error', 'cancelled'].includes(t.status)) return { success: false };

    const idx = this.#queue.indexOf(taskId);
    if (idx !== -1) {
      this.#queue.splice(idx, 1);
      this.#update(taskId, 'cancelled', t.progress);
      return { success: true };
    }
    if (this.#activeTaskId === taskId) {
      this.#worker.postMessage({ type: 'cancel', taskId });
      return { success: true };
    }
    return { success: false };
  }

  async getResult(taskId) {
    const t = this.#requireTask(taskId);
    if (t.status === 'error') throw new Error(t.error || 'Задача завершилась с ошибкой');
    if (t.status !== 'done') throw new Error(`Результат ещё не готов (статус: ${t.status})`);
    return t.resultBlob;
  }

  /** Авторская доработка: коэффициенты, предложенные моделью для этой задачи. */
  getSuggestedCoefficients(taskId) {
    const t = this.#requireTask(taskId);
    return t.coefficients ? { ...t.coefficients } : null;
  }

  /**
   * Авторская доработка: пересчитывает результат с новыми коэффициентами
   * без повторного обращения к модели - мгновенно, на клиенте.
   */
  async renderWithAdjustment(taskId, { brightness, contrast, saturation }) {
    const t = this.#requireTask(taskId);
    if (!t.originalBitmap) throw new Error('Нет исходного изображения для перерасчёта (задача ещё не завершена)');

    const canvas = new OffscreenCanvas(t.originalBitmap.width, t.originalBitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.filter = `brightness(${brightness * 100}%) contrast(${contrast * 100}%) saturate(${saturation * 100}%)`;
    ctx.drawImage(t.originalBitmap, 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
    t.resultBlob = blob;
    return blob;
  }

  // --------------------------- внутреннее ---------------------------

  #hasSupportedExtension(file) {
    const name = (file.name || '').toLowerCase();
    return SUPPORTED_EXT.some((ext) => name.endsWith(ext)) || SUPPORTED_EXT.some((ext) => (file.type || '').includes(ext.slice(1)));
  }

  #requireTask(taskId) {
    const t = this.#tasks.get(taskId);
    if (!t) throw new Error(`Неизвестная задача: ${taskId}`);
    return t;
  }

  #update(taskId, status, progress) {
    const t = this.#tasks.get(taskId);
    if (!t) return;
    t.status = status;
    t.progress = progress;
    this.#emit(taskId);
  }

  #emit(taskId) {
    const t = this.#tasks.get(taskId);
    if (!t) return;
    this.dispatchEvent(new CustomEvent('statuschange', { detail: { taskId, status: t.status, progress: t.progress } }));
  }

  async #advanceQueue() {
    if (this.#activeTaskId) return;
    const taskId = this.#queue.shift();
    if (!taskId) return;

    const t = this.#tasks.get(taskId);
    this.#activeTaskId = taskId;
    this.#update(taskId, 'decoding', 5);

    this.#watchdog = setTimeout(() => {
      if (this.#activeTaskId === taskId) {
        this.#worker.postMessage({ type: 'cancel', taskId });
        t.error = 'Превышено максимальное время обработки (30с)';
        this.#update(taskId, 'error', t.progress);
        this.#finishActive();
      }
    }, MAX_PROCESSING_MS);

    try {
      let payload = t.file;
      const name = (t.file.name || '').toLowerCase();
      if (name.endsWith('.heic') || name.endsWith('.heif') || t.file.type === 'image/heic') {
        const { toPng } = await import('./heicLoader.js');
        payload = await toPng(t.file);
      }
      const buffer = await payload.arrayBuffer();
      this.#worker.postMessage({ type: 'process', taskId, buffer, mime: payload.type || 'image/jpeg' }, [buffer]);
    } catch (err) {
      t.error = err.message || String(err);
      this.#update(taskId, 'error', t.progress);
      this.#finishActive();
    }
  }

  #finishActive() {
    clearTimeout(this.#watchdog);
    this.#activeTaskId = null;
    this.#advanceQueue();
  }

  #handleWorkerMessage(msg) {
    const t = this.#tasks.get(msg.taskId);
    if (!t) return;

    switch (msg.type) {
      case 'progress':
        this.#update(msg.taskId, msg.status, msg.progress);
        break;
      case 'result':
        t.resultBlob = msg.blob;
        t.coefficients = msg.coefficients;
        t.originalBitmap = msg.originalBitmap;
        this.#update(msg.taskId, 'done', 100);
        this.#finishActive();
        break;
      case 'error':
        t.error = msg.error;
        this.#update(msg.taskId, 'error', t.progress);
        this.#finishActive();
        break;
      case 'cancelled':
        this.#update(msg.taskId, 'cancelled', t.progress);
        this.#finishActive();
        break;
      default:
        break;
    }
  }
}

export { MAX_MEGAPIXELS, MAX_PROCESSING_MS };
