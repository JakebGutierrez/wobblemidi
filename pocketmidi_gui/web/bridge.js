// bridge.js — the ONLY file that knows about window.pywebview.
// A future web demo replaces this file with a fetch-based implementation of the
// same interface; nothing else in the front-end may touch pywebview directly.

const bridge = (() => {
  let resolveReady;
  const ready = new Promise((res) => { resolveReady = res; });

  if (window.pywebview && window.pywebview.api) {
    resolveReady();
  } else {
    window.addEventListener("pywebviewready", () => resolveReady());
  }

  async function call(method, ...args) {
    await ready;
    return window.pywebview.api[method](...args);
  }

  return {
    ready,
    ping: () => call("ping"),
    // Phase 1+: loadMidi, humanise, reroll, exportMidi, getState
  };
})();

window.bridge = bridge;
