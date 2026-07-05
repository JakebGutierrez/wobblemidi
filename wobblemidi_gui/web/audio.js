// audio.js — WebAudio playback: synthesized 808 kit (zero assets), lookahead
// scheduler, dual buses for live original/humanised A/B.
//
// Timing contract (design-review fixes, do not regress):
//  * ALL scheduling is anchored to one immutable transport start:
//      when = playStartAudioTime + (hit.ms - playStartSongMs) / 1000
//    The playhead is derived from ctx.currentTime against the same anchor —
//    never accumulated from rAF/setInterval deltas (that drifts on long songs).
//  * Both version buses are always fully scheduled; A/B is a short automated
//    crossfade (cancelScheduledValues → setValueAtTime → linearRamp) placed at
//    ctx.currentTime + GUARD so automation lands on the audio thread, never a
//    raw gain.value flip at the UI click instant.
//  * The 25 ms / 200 ms lookahead is a minimum; every voice starts its gain at
//    zero with a short attack and decays to silence before stop() — abrupt
//    cuts click, especially noise voices (hats/snare).
//  * Sample kits (if ever bundled) must schedule with per-sample onset
//    compensation — see KIT_ONSETS_MS.

const audioEngine = (() => {
  const LOOKAHEAD_MS = 25;        // scheduler tick
  const HORIZON_S = 0.2;          // schedule-ahead window (minimum, generous for WKWebView)
  const GUARD_S = 0.006;          // UI thread → audio thread automation guard
  const XFADE_S = 0.012;          // A/B crossfade
  const START_DELAY_S = 0.09;     // gap between pressing play and bar 1
  const TAIL_MS = 400;            // run-out after the last hit

  let ctx = null;
  let master = null;
  let buses = null;               // {original: GainNode, humanised: GainNode}
  let noiseBuf = null;

  let songs = { original: null, humanised: null };  // sorted hit arrays
  let durationMs = 0;
  let audible = "humanised";
  let playing = false;
  let playStartAudioTime = 0;     // immutable per play()
  let playStartSongMs = 0;        // immutable per play()
  let schedIdx = { original: 0, humanised: 0 };
  let timer = null;
  let raf = null;
  let voices = new Set();         // {src} — every started source, for hard stop
  let lastOpenHat = { original: null, humanised: null };  // per-bus choke target

  let onPlayhead = null;          // (ms|null) => void
  let onTransport = null;         // (playing) => void

  // ---- graph -----------------------------------------------------------------

  function ensureCtx() {
    if (ctx) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    master = ctx.createGain();
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -12;
    comp.knee.value = 18;
    comp.ratio.value = 5;
    comp.attack.value = 0.002;
    comp.release.value = 0.12;
    master.connect(comp);
    comp.connect(ctx.destination);
    buses = { original: ctx.createGain(), humanised: ctx.createGain() };
    buses.original.connect(master);
    buses.humanised.connect(master);
    noiseBuf = makeNoise();
  }

  function makeNoise() {
    const len = 2 * ctx.sampleRate;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }

  // ---- 808 voice bank ----------------------------------------------------------
  // Every voice: gain starts at 0, short attack, exponential decay to silence,
  // sources stopped only after the envelope has closed.

  function env(when, peak, decayS, attackS = 0.0015) {
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(peak, when + attackS);
    g.gain.exponentialRampToValueAtTime(0.0008, when + attackS + decayS);
    g.gain.linearRampToValueAtTime(0, when + attackS + decayS + 0.008);
    return g;
  }

  function osc(type, freq, when, stopAt) {
    const o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(freq, when);
    o.start(when);
    o.stop(stopAt);
    voices.add(o);
    o.onended = () => voices.delete(o);
    return o;
  }

  function noise(when, stopAt) {
    const s = ctx.createBufferSource();
    s.buffer = noiseBuf;
    s.loop = true;
    s.start(when, Math.random() * 1.0);
    s.stop(stopAt);
    voices.add(s);
    s.onended = () => voices.delete(s);
    return s;
  }

  function vKick(when, vel, dest) {
    const end = when + 0.5;
    const o = osc("sine", 165, when, end);
    o.frequency.exponentialRampToValueAtTime(48, when + 0.09);
    const g = env(when, vel * 1.0, 0.42);
    o.connect(g); g.connect(dest);
    // beater click
    const n = noise(when, when + 0.03);
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.frequency.value = 3200; bp.Q.value = 1.2;
    const ng = env(when, vel * 0.32, 0.012, 0.0008);
    n.connect(bp); bp.connect(ng); ng.connect(dest);
  }

  function vSnare(when, vel, dest) {
    const end = when + 0.35;
    for (const [f, p] of [[185, 0.5], [330, 0.32]]) {
      const o = osc("triangle", f, when, end);
      const g = env(when, vel * p, 0.11);
      o.connect(g); g.connect(dest);
    }
    const n = noise(when, end);
    const hp = ctx.createBiquadFilter();
    hp.type = "highpass"; hp.frequency.value = 900;
    const g = env(when, vel * 0.85, 0.17, 0.001);
    n.connect(hp); hp.connect(g); g.connect(dest);
  }

  const METAL_FREQS = [263, 400, 421, 474, 587, 845];

  function metalBank(when, stopAt, hpFreq, bpFreq) {
    const hp = ctx.createBiquadFilter();
    hp.type = "highpass"; hp.frequency.value = hpFreq;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.frequency.value = bpFreq; bp.Q.value = 0.9;
    for (const f of METAL_FREQS) osc("square", f, when, stopAt).connect(bp);
    bp.connect(hp);
    return hp;
  }

  function vHat(when, vel, dest, open) {
    const decay = open ? 0.38 : 0.048;
    const end = when + decay + 0.1;
    const bank = metalBank(when, end, 7200, 10000);
    const g = env(when, vel * (open ? 0.5 : 0.48), decay, 0.001);
    bank.connect(g); g.connect(dest);
    return g;  // choke handle
  }

  function vTom(when, vel, dest, f0, f1, decay) {
    const end = when + decay + 0.15;
    const o = osc("sine", f0, when, end);
    o.frequency.exponentialRampToValueAtTime(f1, when + 0.13);
    const g = env(when, vel * 0.9, decay);
    o.connect(g); g.connect(dest);
    const n = noise(when, when + 0.02);
    const ng = env(when, vel * 0.12, 0.01, 0.0008);
    n.connect(ng); ng.connect(dest);
  }

  function vCrash(when, vel, dest) {
    const end = when + 1.7;
    const n = noise(when, end);
    const hp = ctxFilter("highpass", 4200);
    const g = env(when, vel * 0.55, 1.35, 0.001);
    n.connect(hp); hp.connect(g); g.connect(dest);
    const bank = metalBank(when, end, 5200, 8600);
    const bg = env(when, vel * 0.2, 0.9, 0.001);
    bank.connect(bg); bg.connect(dest);
  }

  function vRide(when, vel, dest) {
    const end = when + 1.1;
    const bank = metalBank(when, end, 6300, 9000);
    const bg = env(when, vel * 0.28, 0.75, 0.001);
    bank.connect(bg); bg.connect(dest);
    const ping = osc("sine", 1050, when, end);
    const pg = env(when, vel * 0.16, 0.32);
    ping.connect(pg); pg.connect(dest);
    const n = noise(when, end);
    const hp = ctxFilter("highpass", 7600);
    const ng = env(when, vel * 0.1, 0.6, 0.001);
    n.connect(hp); hp.connect(ng); ng.connect(dest);
  }

  function ctxFilter(type, freq) {
    const f = ctx.createBiquadFilter();
    f.type = type; f.frequency.value = freq;
    return f;
  }

  // lane → voice + loudness trim (kit balance)
  const LANE_VOICES = {
    kick:         (t, v, d) => vKick(t, v, d),
    snare:        (t, v, d) => vSnare(t, v, d),
    hihat_closed: (t, v, d) => vHat(t, v, d, false),
    hihat_open:   (t, v, d) => vHat(t, v, d, true),
    tom_high:     (t, v, d) => vTom(t, v, d, 176, 98, 0.26),
    tom_mid:      (t, v, d) => vTom(t, v, d, 136, 76, 0.3),
    tom_low:      (t, v, d) => vTom(t, v, d, 102, 56, 0.36),
    crash:        (t, v, d) => vCrash(t, v, d),
    ride:         (t, v, d) => vRide(t, v, d),
  };

  const LANE_TRIM = {
    kick: 1.0, snare: 0.95, hihat_closed: 0.62, hihat_open: 0.66,
    tom_high: 0.85, tom_mid: 0.85, tom_low: 0.88, crash: 0.72, ride: 0.55,
  };

  // Per-sample onset compensation for a future sample kit: schedule at
  // when - KIT_ONSETS_MS[lane]/1000 so perceived onsets match the MIDI time.
  // The synth kit has zero-latency onsets by construction.
  const KIT_ONSETS_MS = {};

  function velGain(lane, velocity) {
    return Math.pow(velocity / 127, 1.7) * (LANE_TRIM[lane] || 0.8);
  }

  function trigger(lane, when, velocity, busName) {
    const make = LANE_VOICES[lane];
    if (!make) return;
    const onset = (KIT_ONSETS_MS[lane] || 0) / 1000;
    const bus = buses[busName];
    if (lane === "hihat_closed" && lastOpenHat[busName]) {
      // 808 hat choke: a closed hat shuts the ringing open hat at its own hit time
      const oh = lastOpenHat[busName];
      const t = Math.max(when - onset, ctx.currentTime + GUARD_S);
      oh.gain.cancelScheduledValues(t);
      oh.gain.setValueAtTime(oh.gain.value, t);
      oh.gain.linearRampToValueAtTime(0, t + 0.006);
      lastOpenHat[busName] = null;
    }
    const handle = make(when - onset, velGain(lane, velocity), bus);
    if (lane === "hihat_open") lastOpenHat[busName] = handle;
  }

  // ---- transport -----------------------------------------------------------------

  function msToWhen(ms) {
    return playStartAudioTime + (ms - playStartSongMs) / 1000;
  }

  function schedulerTick() {
    const limit = ctx.currentTime + HORIZON_S;
    for (const name of ["original", "humanised"]) {
      const hits = songs[name];
      if (!hits) continue;
      let i = schedIdx[name];
      while (i < hits.length && msToWhen(hits[i].ms) < limit) {
        const when = msToWhen(hits[i].ms);
        if (when >= ctx.currentTime - 0.01) {
          trigger(hits[i].lane, Math.max(when, ctx.currentTime + 0.002), hits[i].velocity, name);
        }
        i++;
      }
      schedIdx[name] = i;
    }
  }

  function playheadLoop() {
    if (!playing) return;
    const ms = playStartSongMs + (ctx.currentTime - playStartAudioTime) * 1000;
    if (onPlayhead) onPlayhead(Math.max(ms, playStartSongMs));
    if (ms > durationMs + TAIL_MS) { stop(); return; }
    raf = requestAnimationFrame(playheadLoop);
  }

  function lowerIndex(hits, ms) {
    let lo = 0, hi = hits.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (hits[mid].ms < ms) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function play(fromMs = 0) {
    if (playing || (!songs.original && !songs.humanised)) return;
    ensureCtx();
    if (ctx.state === "suspended") ctx.resume();

    playing = true;
    playStartSongMs = Math.max(0, fromMs);
    playStartAudioTime = ctx.currentTime + START_DELAY_S;

    const want = songs[audible] ? audible : "original";
    audible = want;
    for (const name of ["original", "humanised"]) {
      buses[name].gain.cancelScheduledValues(ctx.currentTime);
      buses[name].gain.setValueAtTime(name === want ? 1 : 0, ctx.currentTime);
      schedIdx[name] = songs[name] ? lowerIndex(songs[name], playStartSongMs) : 0;
      lastOpenHat[name] = null;
    }
    master.gain.cancelScheduledValues(ctx.currentTime);
    master.gain.setValueAtTime(0, ctx.currentTime);
    master.gain.linearRampToValueAtTime(1, ctx.currentTime + 0.01);

    schedulerTick();
    timer = setInterval(schedulerTick, LOOKAHEAD_MS);
    raf = requestAnimationFrame(playheadLoop);
    if (onTransport) onTransport(true);
  }

  function stop() {
    if (!playing) return;
    playing = false;
    clearInterval(timer);
    timer = null;
    if (raf) cancelAnimationFrame(raf);

    // ramp out, then kill sources — never cut noise voices abruptly
    const t = ctx.currentTime + GUARD_S;
    master.gain.cancelScheduledValues(ctx.currentTime);
    master.gain.setValueAtTime(master.gain.value, t);
    master.gain.linearRampToValueAtTime(0, t + 0.015);
    const doomed = [...voices];
    setTimeout(() => {
      for (const src of doomed) { try { src.stop(); } catch (e) { /* already stopped */ } }
    }, 60);

    if (onPlayhead) onPlayhead(null);
    if (onTransport) onTransport(false);
  }

  function setAudible(name) {
    if (!songs[name]) return;
    audible = name;
    if (!ctx || !playing) return;
    // guarded, automated crossfade — both buses keep running
    const now = ctx.currentTime;
    const t = now + GUARD_S;
    for (const b of ["original", "humanised"]) {
      const g = buses[b].gain;
      const target = b === name ? 1 : 0;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, t);
      g.linearRampToValueAtTime(target, t + XFADE_S);
    }
  }

  function setSongs(original, humanised, dur) {
    const wasPlaying = playing;
    if (playing) stop();
    songs.original = original ? [...original].sort((a, b) => a.ms - b.ms) : null;
    songs.humanised = humanised ? [...humanised].sort((a, b) => a.ms - b.ms) : null;
    durationMs = dur || 0;
    if (!songs[audible]) audible = songs.humanised ? "humanised" : "original";
    return wasPlaying;
  }

  return {
    play,
    stop,
    setSongs,
    setAudible,
    getAudible: () => audible,
    isPlaying: () => playing,
    onPlayhead: (fn) => { onPlayhead = fn; },
    onTransport: (fn) => { onTransport = fn; },
    hasSong: () => !!(songs.original || songs.humanised),
    hasBoth: () => !!(songs.original && songs.humanised),
    ctxState: () => (ctx ? ctx.state : "none"),   // diagnostics: "running" = audibly rendering
  };
})();

window.audioEngine = audioEngine;
