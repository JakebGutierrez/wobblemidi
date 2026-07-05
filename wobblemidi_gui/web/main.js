// main.js — top-level UI wiring: knobs → params, buttons → bridge, transport → screen.

const LANE_NAMES = {
  kick: "KICK", snare: "SNARE", hihat_closed: "HAT CL", hihat_open: "HAT OP",
  tom_high: "TOM HI", tom_mid: "TOM MD", tom_low: "TOM LO",
  crash: "CRASH", ride: "RIDE",
};

const state = {
  loaded: false,
  fileName: null,
  params: { intensity: 0.35, tightness: 0.4, lean: 0.0, all_channels: false },
  laneIntensity: {},     // lane → absolute intensity override
  selectedLane: null,    // lane-select scope for the INTENSITY knob
  busy: false,
  humanised: false,      // a render exists
  canUndo: false,
  playFrom: 0,
  bars: [],
  durationMs: 0,
};

let knobIntensity = null;
let knobTightness = null;
let sliderLean = null;

const el = (id) => document.getElementById(id);

function toast(message, ms = 3600) {
  const t = el("toast");
  t.textContent = message;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), ms);
}
window.toast = toast;

function renderParams() {
  return { ...state.params, lane_intensity: { ...state.laneIntensity } };
}

// ---- transport readout ---------------------------------------------------------

function fmtTime(ms) {
  const s = Math.max(0, ms) / 1000;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

function barIndexAt(ms) {
  const bars = state.bars;
  let lo = 0, hi = bars.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bars[mid].ms <= ms + 1e-6) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

function updateReadout(ms) {
  if (!state.loaded) {
    el("readout").textContent = "BAR – / – · 0:00.0";
    return;
  }
  const clamped = Math.min(Math.max(ms, 0), state.durationMs);
  el("readout").textContent =
    `BAR ${barIndexAt(clamped) + 1} / ${state.bars.length} · ${fmtTime(clamped)}`;
}

// ---- audible version (chips + screen + engine agree) -----------------------------

function setAudibleVersion(v) {
  audioEngine.setAudible(v);
  grooveScreen.setAudible(v);
  const hearing = audioEngine.getAudible();
  el("vchip-orig").classList.toggle("active", hearing === "original");
  el("vchip-hum").classList.toggle("active", hearing === "humanised");
  el("btn-ab").classList.toggle("lit", hearing === "original");
}

// ---- lane scope ------------------------------------------------------------------

function refreshLaneMarks() {
  grooveScreen.setLaneMarks(state.selectedLane, Object.keys(state.laneIntensity));
}

function setLaneScope(lane) {
  state.selectedLane = lane;
  const scoped = lane !== null;
  el("lbl-intensity").textContent =
    scoped ? `INTENSITY · ${LANE_NAMES[lane] || lane.toUpperCase()}` : "INTENSITY";
  el("btn-kit").classList.toggle("visible", scoped);
  el("knob-tightness").classList.toggle("kitwide-dim", scoped);
  el("slider-lean").classList.toggle("kitwide-dim", scoped);
  knobIntensity.set(
    scoped ? (state.laneIntensity[lane] ?? state.params.intensity)
           : state.params.intensity,
    false,
  );
  refreshLaneMarks();
}

// ---- button state ------------------------------------------------------------------

function updateButtons() {
  const { loaded, busy, humanised, canUndo } = state;
  el("btn-load").disabled = busy;
  el("btn-humanise").disabled = !loaded || busy;
  el("btn-undo").disabled = !canUndo || busy;
  el("btn-play").disabled = !loaded || busy;
  el("btn-ab").disabled = !humanised || busy;
  el("btn-export").disabled = !humanised || busy;
  el("vchip-orig").disabled = !humanised || busy;
  el("vchip-hum").disabled = !humanised || busy;
  for (const id of ["btn-tostart", "btn-barback", "btn-barfwd",
                    "btn-zoomin", "btn-zoomout", "btn-fit"]) {
    el(id).disabled = !loaded;
  }
  el("btn-humanise").classList.toggle("armed", loaded && !humanised && !busy);
}

function setBusy(busy) {
  state.busy = busy;
  el("btn-humanise").classList.toggle("busy", busy);
  updateButtons();
}

function updateTransport(playing) {
  el("btn-play").textContent = playing ? "STOP" : "PLAY";
  el("btn-play").classList.toggle("lit", playing);
  if (!playing) {
    grooveScreen.setPlayhead(null, false);
    updateReadout(state.playFrom);
  }
}

// ---- warnings ------------------------------------------------------------------------

function describeWarnings(res) {
  for (const w of res.warnings || []) {
    if (w === "no_drum_hits_channel10") {
      const n = res.original.other_channel_drum_hits;
      toast(`No drum hits on channel 10 — ${n} drum-range note${n === 1 ? "" : "s"} on ` +
            `other channels. ALL CH switched on for you; press HUMANISE to include them.`, 6000);
      el("toggle-allch").classList.add("on");
      state.params.all_channels = true;
    } else if (w === "no_drum_hits") {
      toast("No drum hits found in this file.");
    }
  }
}

// ---- actions ----------------------------------------------------------------------------

function applyLoad(res) {
  audioEngine.stop();
  state.loaded = true;
  state.fileName = res.file_name;
  state.humanised = false;
  state.canUndo = false;
  state.playFrom = 0;
  state.laneIntensity = {};
  state.bars = res.original.bars;
  state.durationMs = res.original.duration_ms;
  setLaneScope(null);
  el("screen-empty").classList.remove("visible");
  grooveScreen.setData(res.original, null);
  grooveScreen.setStartMarker(null);
  audioEngine.setSongs(res.original.hits, null, res.original.duration_ms);
  setAudibleVersion("original");
  el("vchip-orig").classList.remove("active");   // no comparison yet — chips stay neutral
  el("vchip-hum").classList.remove("active");
  describeWarnings(res);
  updateButtons();
  updateReadout(0);
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

function applyRender(res) {
  state.humanised = true;
  state.canUndo = !!res.can_undo;
  grooveScreen.setData(res.original, res.humanised);
  audioEngine.setSongs(res.original.hits, res.humanised.hits,
                       res.original.duration_ms);
  setAudibleVersion("humanised");
  refreshLaneMarks();
}

async function doHumanise() {
  if (!state.loaded || state.busy) return;
  audioEngine.stop();
  setBusy(true);
  try {
    const res = await bridge.humanise(renderParams());
    if (!res.ok) { toast(`Humanise failed: ${res.error}`); return; }
    applyRender(res);
    const hero = el("btn-humanise");
    hero.classList.remove("flash");
    void hero.offsetWidth;   // restart the animation
    hero.classList.add("flash");
  } finally {
    setBusy(false);
  }
}

async function doUndo() {
  if (!state.canUndo || state.busy) return;
  audioEngine.stop();
  setBusy(true);
  try {
    const res = await bridge.undo();
    if (!res.ok) { toast(res.error); return; }
    // restore the settings that produced the restored render
    const p = res.params || {};
    state.params.intensity = p.intensity ?? state.params.intensity;
    state.params.tightness = p.tightness ?? state.params.tightness;
    state.params.lean = p.lean ?? state.params.lean;
    state.params.all_channels = !!p.all_channels;
    state.laneIntensity = { ...(p.lane_intensity || {}) };
    knobTightness.set(state.params.tightness, false);
    sliderLean.set(state.params.lean, false);
    el("toggle-allch").classList.toggle("on", state.params.all_channels);
    setLaneScope(state.selectedLane);   // re-sync knob value + marks under new params
    applyRender(res);
    toast("Restored previous render (UNDO again = redo).");
  } finally {
    setBusy(false);
  }
}

function doPlay() {
  if (!state.loaded || state.busy) return;
  if (audioEngine.isPlaying()) audioEngine.stop();
  else audioEngine.play(state.playFrom);
}

function doAB() {
  if (!audioEngine.hasBoth()) return;
  setAudibleVersion(audioEngine.getAudible() === "humanised" ? "original" : "humanised");
}

function seekTo(ms, restart = true) {
  state.playFrom = Math.min(Math.max(ms, 0), state.durationMs);
  grooveScreen.setStartMarker(state.playFrom);
  updateReadout(state.playFrom);
  if (restart && audioEngine.isPlaying()) {
    audioEngine.stop();
    audioEngine.play(state.playFrom);
  }
}

function doToStart() {
  if (!state.loaded) return;
  seekTo(0);
  grooveScreen.setStartMarker(null);
  grooveScreen.setView(0);   // scroll home, keep the current zoom
}

function stepBar(direction) {
  if (!state.loaded || !state.bars.length) return;
  const idx = barIndexAt(state.playFrom);
  const next = Math.min(Math.max(idx + direction, 0), state.bars.length - 1);
  seekTo(state.bars[next].ms);
}

async function doExport() {
  if (!state.humanised || state.busy) return;
  const res = await bridge.exportMidi();
  if (res.cancelled) return;
  if (!res.ok) { toast(`Export failed: ${res.error}`); return; }
  toast(`Exported → ${res.path}`);
}

// ---- boot ---------------------------------------------------------------------------------

(async () => {
  grooveScreen.init(el("groove-screen"), el("minimap"));

  knobIntensity = makeKnob({
    mount: el("knob-intensity"), min: 0, max: 1, value: 0.35, def: 0.35,
    onInput: (v) => {
      if (state.selectedLane) {
        state.laneIntensity[state.selectedLane] = v;
        refreshLaneMarks();
      } else {
        state.params.intensity = v;
      }
    },
    onReset: () => {
      if (state.selectedLane) {
        // clear this lane's override → it follows the master knob again
        delete state.laneIntensity[state.selectedLane];
        knobIntensity.set(state.params.intensity, false);
        refreshLaneMarks();
      } else {
        knobIntensity.set(0.35);
      }
    },
  });
  knobTightness = makeKnob({
    mount: el("knob-tightness"), min: 0, max: 0.95, value: 0.4, def: 0.4,
    onInput: (v) => { state.params.tightness = v; },
  });
  sliderLean = makeSlider({
    mount: el("slider-lean"), min: -1, max: 1, value: 0, def: 0, detent: 0.06,
    fmt: (v) => {
      if (v === 0) return "NEUTRAL";
      const pct = Math.round(Math.abs(v) * 100);
      return v > 0 ? `PUSH ${pct}%` : `MIRROR ${pct}%`;
    },
    onInput: (v) => { state.params.lean = v; },
  });
  el("toggle-allch").addEventListener("click", () => {
    const on = el("toggle-allch").classList.toggle("on");
    state.params.all_channels = on;
    toast(on ? "All channels: drum-range notes everywhere get humanised (next render)."
             : "Channel 10 only (standard drum channel).");
  });

  el("btn-load").addEventListener("click", doLoad);
  el("btn-humanise").addEventListener("click", doHumanise);
  el("btn-undo").addEventListener("click", doUndo);
  el("btn-play").addEventListener("click", doPlay);
  el("btn-ab").addEventListener("click", doAB);
  el("btn-export").addEventListener("click", doExport);
  el("btn-tostart").addEventListener("click", doToStart);
  el("btn-barback").addEventListener("click", () => stepBar(-1));
  el("btn-barfwd").addEventListener("click", () => stepBar(1));
  el("btn-zoomin").addEventListener("click", () => grooveScreen.zoomBy(0.65));
  el("btn-zoomout").addEventListener("click", () => grooveScreen.zoomBy(1.55));
  el("btn-fit").addEventListener("click", () => grooveScreen.fit());
  el("vchip-orig").addEventListener("click", () => setAudibleVersion("original"));
  el("vchip-hum").addEventListener("click", () => setAudibleVersion("humanised"));
  el("btn-kit").addEventListener("click", () => setLaneScope(null));

  el("btn-help").addEventListener("click", () => el("help-overlay").classList.add("visible"));
  el("btn-help-close").addEventListener("click", () => el("help-overlay").classList.remove("visible"));
  el("help-overlay").addEventListener("click", (e) => {
    if (e.target === el("help-overlay")) el("help-overlay").classList.remove("visible");
  });

  // buttons keep focus after click; blur so space stays the play/stop key
  for (const b of document.querySelectorAll("button")) {
    b.addEventListener("click", () => b.blur());
  }

  audioEngine.onPlayhead((ms) => {
    if (ms === null) return;
    grooveScreen.setPlayhead(ms, true);
    updateReadout(ms);
  });
  audioEngine.onTransport(updateTransport);

  grooveScreen.onPick((ms) => seekTo(ms));
  grooveScreen.onLane((lane) => {
    setLaneScope(lane === state.selectedLane ? null : lane);
  });
  grooveScreen.onScrub(
    (ms) => {   // move: follow with marker + readout, no audio churn
      grooveScreen.setStartMarker(ms);
      updateReadout(ms);
    },
    (ms) => seekTo(ms),   // release: commit (restarts if playing)
  );

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); doPlay(); }
    else if (e.code === "KeyA" && !e.metaKey && !e.ctrlKey) doAB();
    else if (e.code === "Home") doToStart();
    else if (e.code === "ArrowLeft") stepBar(-1);
    else if (e.code === "ArrowRight") stepBar(1);
    else if (e.code === "Escape") {
      el("help-overlay").classList.remove("visible");
      if (state.selectedLane) setLaneScope(null);
    }
  });

  updateButtons();
  updateReadout(0);

  try {
    if ((await bridge.ping()) === "pong") el("power-led").classList.add("on");
    const status = await bridge.getStatus();
    if (!status.ok) {
      el("power-led").classList.remove("on");
      el("power-led").classList.add("err");
      toast(status.error);
      return;
    }
    const auto = await bridge.autoload();
    if (auto && auto.ok) applyLoad(auto);
    else if (auto && auto.error) toast(`Can't load: ${auto.error}`);
  } catch (err) {
    console.error("backend bridge failed:", err);
  }
})();
