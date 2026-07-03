// main.js — top-level UI wiring: bridge calls, button states, screen data flow.
// Knob-driven params arrive in Phase 3; until then HUMANISE uses the defaults below.

const state = {
  loaded: false,
  fileName: null,
  params: { intensity: 0.35, tightness: 0.4, push: false, all_channels: false },
  seed: null,
  busy: false,
};

const el = (id) => document.getElementById(id);

function toast(message, ms = 3600) {
  const t = el("toast");
  t.textContent = message;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), ms);
}
window.toast = toast;

function setBusy(busy) {
  state.busy = busy;
  el("btn-humanise").classList.toggle("busy", busy);
  for (const id of ["btn-load", "btn-humanise", "btn-export"]) {
    el(id).disabled = busy || (id !== "btn-load" && !state.loaded);
  }
  if (!busy) el("btn-export").disabled = !state.seed;
}

function describeWarnings(res) {
  for (const w of res.warnings || []) {
    if (w === "no_drum_hits_channel10") {
      const n = res.original.other_channel_drum_hits;
      toast(`No drum hits on channel 10 — ${n} drum-range note${n === 1 ? "" : "s"} found on other channels.`);
    } else if (w === "no_drum_hits") {
      toast("No drum hits found in this file.");
    }
  }
}

function applyLoad(res) {
  state.loaded = true;
  state.fileName = res.file_name;
  state.seed = null;
  el("screen-empty").classList.remove("visible");
  grooveScreen.setData(res.original, null);
  el("btn-humanise").disabled = false;
  el("btn-export").disabled = true;
  describeWarnings(res);
  toast(`Loaded ${res.file_name} — ${res.original.hits.length} hits, ` +
        `${res.original.bars.length} bars @ ${res.original.bpm} BPM`);
}

async function doLoad() {
  if (state.busy) return;
  const res = await bridge.openMidi();
  if (res.cancelled) return;
  if (!res.ok) { toast(`Can't load: ${res.error}`); return; }
  applyLoad(res);
}

async function doHumanise() {
  if (!state.loaded || state.busy) return;
  setBusy(true);
  try {
    const res = await bridge.humanise(state.params);
    if (!res.ok) { toast(`Humanise failed: ${res.error}`); return; }
    state.seed = res.seed;
    grooveScreen.setData(res.original, res.humanised);
  } finally {
    setBusy(false);
  }
}

async function doExport() {
  if (!state.seed || state.busy) return;
  const res = await bridge.exportMidi();
  if (res.cancelled) return;
  if (!res.ok) { toast(`Export failed: ${res.error}`); return; }
  toast(`Exported → ${res.path}`);
}

// ---- boot -------------------------------------------------------------------

(async () => {
  grooveScreen.init(el("groove-screen"), el("minimap"));

  el("btn-load").addEventListener("click", doLoad);
  el("btn-humanise").addEventListener("click", doHumanise);
  el("btn-export").addEventListener("click", doExport);

  try {
    if ((await bridge.ping()) === "pong") el("power-led").classList.add("on");
    const status = await bridge.getStatus();
    if (!status.ok) { toast(status.error); return; }
    const auto = await bridge.autoload();
    if (auto && auto.ok) applyLoad(auto);
    else if (auto && auto.error) toast(`Can't load: ${auto.error}`);
  } catch (err) {
    console.error("backend bridge failed:", err);
  }
})();
