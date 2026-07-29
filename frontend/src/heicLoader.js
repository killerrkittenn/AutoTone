/**
 * heicLoader.js — конвертация HEIC/HEIF в PNG перед обработкой,
 * т.к. большинство браузеров (кроме Safari) не декодируют HEIC нативно.
 * Библиотека подключается динамически, только если файл реально HEIC,
 * чтобы не раздувать базовый бандл.
 */
const PKG = 'heic2any';
const VERSION = '0.0.4';

let loaderPromise = null;

function load() {
  if (!loaderPromise) {
    loaderPromise = import(`https://esm.sh/${PKG}@${VERSION}`).then((m) => m.default || m);
  }
  return loaderPromise;
}

export async function toPng(heicBlob) {
  const heic2any = await load();
  const result = await heic2any({ blob: heicBlob, toType: 'image/png', quality: 1 });
  return Array.isArray(result) ? result[0] : result;
}
