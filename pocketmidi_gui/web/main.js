// main.js — top-level UI wiring: knobs → params, buttons → bridge, transport → screen.

const state = {
  loaded: false,
  fileName: null,
  params: { intensity: 0.35, tightness: 0.4, push: false, all_channels: false },
  seed: null,
  busy: false,
  humanised: false,   // a render exists
  playFrom: 0,
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

// ---- button / chip state ------------------------------------------------------

function updateButtons() {
  const { loaded, busy, humanised } = state;
  el("btn-load").disabled = busy;
  el("btn-humanise").disabled = !loaded || busy;
  el("btn-newtake").disabled = !loaded || busy;
  el("btn-play").disabled = !loaded || busy;
  el("btn-ab").disabled = !humanised || busy;
  el("btn-export").disabled = !humanised || busy;
}

function setBusy(busy) {
  state.busy = busy;
  el("btn-humanise").classList.toggle("busy", busy);
  updateButtons();
}

function updateHearingChip() {
  const chip = el("hearing-chip");
  const playing = audioEngine.isPlaying();
  chip.classList.toggle("visible", playing);
  if (!playing) return;
  const hearing = audioEngine.getAudible();
  chip.textContent = hearing === "original" ? "● ORIGINAL" : "● HUMANISED";
  chip.classList.toggle("orig", hearing === "original");
  el("btn-ab").classList.toggle("lit", hearing === "original");
}

function updateTransport(playing) {
  el("btn-play").textContent = playing ? "STOP" : "PLAY";
  el("btn-play").classList.toggle("lit", playing);
  if (!playing) {
    el("btn-ab").classList.remove("lit");
    grooveScreen.setPlayhead(null, false);
  }
  updateHearingChip();
}

// ---- warnings -------------------------------------------------------------------

function describeWarnings(res) {
  for (const w of res.warnings || []) {
    if (w === "no_drum_hits_channel10") {
      const n = res.original.other_channel_drum_hits;
      toast(`No drum hits on channel 10 — ${n} drum-range note${n === 1 ? "" : "s"} on ` +
            `other channels. Switch ALL CH on and humanise to include them.`, 6000);
      el("toggle-allch").classList.add("on");
      state.params.all_channels = true;
    } else if (w === "no_drum_hits") {
      toast("No drum hits found in this file.");
    }
  }
}

// ---- actions ---------------------------------------------------------------------

function applyLoad(res) {
  audioEngine.stop();
  state.loaded = true;
  state.fileName = res.file_name;
  state.seed = null;
  state.humanised = false;
  state.playFrom = 0;
  el("screen-empty").classList.remove("visible");
  grooveScreen.setData(res.original, null);
  grooveScreen.setStartMarker(null);
  audioEngine.setSongs(res.original.hits, null, res.original.duration_ms);
  describeWarnings(res);
  updateButtons();
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

async function renderWith(call) {
  if (!state.loaded || state.busy) return;
  audioEngine.stop();
  setBusy(true);
  try {
    const res = await call(state.params);
    if (!res.ok) { toast(`Humanise failed: ${res.error}`); return; }
    state.seed = res.seed;
    state.humanised = true;
    grooveScreen.setData(res.original, res.humanised);
    audioEngine.setSongs(res.original.hits, res.humanised.hits, res.original.duration_ms);
    audioEngine.setAudible("humanised");
  } finally {
    setBusy(false);
  }
}

const doHumanise = () => renderWith((p) => bridge.humanise(p));
const doNewTake = () => renderWith((p) => bridge.reroll(p));

function doPlay() {
  if (!state.loaded || state.busy) return;
  if (audioEngine.isPlaying()) audioEngine.stop();
  else audioEngine.play(state.playFrom);
}

function doAB() {
  if (!audioEngine.hasBoth()) return;
  const next = audioEngine.getAudible() === "humanised" ? "original" : "humanised";
  audioEngine.setAudible(next);
  updateHearingChip();
}

async function doExport() {
  if (!state.humanised || state.busy) return;
  const res = await bridge.exportMidi();
  if (res.cancelled) return;
  if (!res.ok) { toast(`Export failed: ${res.error}`); return; }
  toast(`Exported → ${res.path}`);
}

// ---- boot -------------------------------------------------------------------------

(async () => {
  grooveScreen.init(el("groove-screen"), el("minimap"));

  makeKnob({
    mount: el("knob-intensity"), min: 0, max: 1, value: 0.35, def: 0.35,
    onInput: (v) => { state.params.intensity = v; },
  });
  makeKnob({
    mount: el("knob-tightness"), min: 0, max: 0.95, value: 0.4, def: 0.4,
    onInput: (v) => { state.params.tightness = v; },
  });
  makeToggle({
    mount: el("toggle-push"), value: false,
    onInput: (v) => { state.params.push = v; },
  });
  el("toggle-allch").addEventListener("click", () => {
    const on = el("toggle-allch").classList.toggle("on");
    state.params.all_channels = on;
    toast(on ? "All channels: drum-range notes everywhere get humanised (next render)."
             : "Channel 10 only (standard drum channel).");
  });

  el("btn-load").addEventListener("click", doLoad);
  el("btn-humanise").addEventListener("click", doHumanise);
  el("btn-newtake").addEventListener("click", doNewTake);
  el("btn-play").addEventListener("click", doPlay);
  el("btn-ab").addEventListener("click", doAB);
  el("btn-export").addEventListener("click", doExport);

  audioEngine.onPlayhead((ms) => {
    grooveScreen.setPlayhead(ms, true);
  });
  audioEngine.onTransport(updateTransport);

  grooveScreen.onPick((ms) => {
    state.playFrom = ms;
    grooveScreen.setStartMarker(ms);
    if (audioEngine.isPlaying()) {
      audioEngine.stop();
      audioEngine.play(ms);
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); doPlay(); }
    else if (e.code === "KeyA" && !e.metaKey && !e.ctrlKey) doAB();
  });

  updateButtons();

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
