// src/resizeObserverFix.js

// 1) Swallow the specific ResizeObserver loop warning
/* eslint-disable no-undef */
window.onerror = function (message, source, lineno, colno, error) {
  if (
    typeof message === 'string' &&
    message.includes('ResizeObserver loop completed with undelivered notifications')
  ) {
    return true; // swallow it
  }
  return false;  // let other errors through
};

// 2) Monkey‑patch ResizeObserver to catch any callback errors
;(function () {
  const NativeRO = window.ResizeObserver;
  if (!NativeRO) return;
  window.ResizeObserver = class ResizeObserverNoThrow extends NativeRO {
    constructor(callback) {
      super((entries, observer) => {
        try {
          callback(entries, observer);
        } catch {
          // swallow
        }
      });
    }
  };
})();
