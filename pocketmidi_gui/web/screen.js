// screen.js — the groove screen: drum lanes with a zoomed scrolling window,
// bar/beat ruler, humanised-shift overlays, full-song minimap, playhead.
//
// Interaction: drag = pan · wheel = pan · ⌘/ctrl+wheel = zoom at cursor ·
// double-click = fit whole song · minimap click/drag = jump.

const grooveScreen = (() => {
  const LANE_LABELS = {
    crash: "CRASH", ride: "RIDE", hihat_open: "HAT OP", hihat_closed: "HAT CL",
    tom_high: "TOM HI", tom_mid: "TOM MD", tom_low: "TOM LO",
    snare: "SNARE", kick: "KICK",
  };
  const LANE_COLORS = {
    crash: "#f6ecc8", ride: "#f2e2a0", hihat_open: "#f0d075", hihat_closed: "#e8b73c",
    tom_high: "#f2c96c", tom_mid: "#f0b95c", tom_low: "#eda94e",
    snare: "#f07d33", kick: "#e8503a",
  };
  const RULER_H = 22;
  const GUTTER_W = 78;
  const TAIL_MS = 500;         // breathing room after the last hit
  const MIN_SPAN = 250;        // ms — max zoom in
  const GHOST_ALPHA = 0.30;    // original positions under a humanised overlay

  let canvas, ctx, mmCanvas, mmCtx;
  let original = null;
  let humanised = null;
  let viewStart = 0;           // ms at left edge of the zoom window
  let viewSpan = 8000;         // ms across the zoom window
  let playheadMs = null;
  let follow = false;          // auto-scroll to keep playhead in view
  let dpr = window.devicePixelRatio || 1;

  // ---- helpers -------------------------------------------------------------

  const data = () => humanised || original;
  const durationMs = () => (original ? original.duration_ms + TAIL_MS : 1);

  function clampView() {
    const dur = durationMs();
    viewSpan = Math.min(Math.max(viewSpan, MIN_SPAN), dur);
    viewStart = Math.min(Math.max(viewStart, 0), Math.max(0, dur - viewSpan));
  }

  function cssSize(c) {
    return { w: c.width / dpr, h: c.height / dpr };
  }

  function resizeCanvas(c) {
    const rect = c.parentElement.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    c.width = Math.max(1, Math.round(rect.width * dpr));
    c.height = Math.max(1, Math.round(rect.height * dpr));
  }

  function xOf(ms, w) {
    return GUTTER_W + ((ms - viewStart) / viewSpan) * (w - GUTTER_W);
  }

  function msAt(x, w) {
    return viewStart + ((x - GUTTER_W) / (w - GUTTER_W)) * viewSpan;
  }

  // ---- main screen ----------------------------------------------------------

  function draw() {
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { w, h } = cssSize(canvas);
    ctx.fillStyle = "#0b0e0c";
    ctx.fillRect(0, 0, w, h);
    if (!original) return;
    clampView();

    const d = data();
    const lanes = d.lanes;
    const laneH = (h - RULER_H) / lanes.length;
    const laneY = (i) => RULER_H + i * laneH;
    const laneIdx = {};
    lanes.forEach((l, i) => { laneIdx[l] = i; });

    // lane row shading + separators
    for (let i = 0; i < lanes.length; i++) {
      ctx.fillStyle = i % 2 ? "rgba(255,255,255,0.016)" : "rgba(255,255,255,0.005)";
      ctx.fillRect(GUTTER_W, laneY(i), w - GUTTER_W, laneH);
      ctx.fillStyle = "rgba(255,182,72,0.07)";
      ctx.fillRect(GUTTER_W, laneY(i), w - GUTTER_W, 1);
    }

    // beat + bar gridlines
    for (const b of d.beats) {
      if (b.ms < viewStart - 1 || b.ms > viewStart + viewSpan + 1) continue;
      const x = xOf(b.ms, w);
      ctx.fillStyle = "rgba(255,182,72,0.06)";
      ctx.fillRect(x, RULER_H, 1, h - RULER_H);
    }
    ctx.font = "9px 'Futura', 'Avenir Next', sans-serif";
    d.bars.forEach((b, i) => {
      if (b.ms < viewStart - viewSpan * 0.1 || b.ms > viewStart + viewSpan * 1.1) return;
      const x = xOf(b.ms, w);
      ctx.fillStyle = "rgba(255,182,72,0.18)";
      ctx.fillRect(x, RULER_H, 1, h - RULER_H);
      ctx.fillStyle = "rgba(255,182,72,0.55)";
      ctx.fillText(String(i + 1), x + 4, 14);
    });

    // ruler base line
    ctx.fillStyle = "rgba(255,182,72,0.25)";
    ctx.fillRect(GUTTER_W, RULER_H - 1, w - GUTTER_W, 1);

    // hits — with a humanised render, originals become ghosts underneath
    if (humanised) drawHits(original.hits, laneIdx, laneY, laneH, w, true);
    drawHits(d.hits, laneIdx, laneY, laneH, w, false);

    // gutter: lane labels over a solid strip (drawn last so hits never overlap it)
    ctx.fillStyle = "#0b0e0c";
    ctx.fillRect(0, 0, GUTTER_W, h);
    ctx.fillStyle = "rgba(255,182,72,0.12)";
    ctx.fillRect(GUTTER_W - 1, 0, 1, h);
    ctx.font = "10px 'Futura', 'Avenir Next', sans-serif";
    lanes.forEach((l, i) => {
      ctx.fillStyle = LANE_COLORS[l] || "#e8b73c";
      ctx.globalAlpha = 0.85;
      ctx.fillText(LANE_LABELS[l] || l.toUpperCase(), 10, laneY(i) + laneH / 2 + 3.5);
      ctx.globalAlpha = 1;
    });

    // playhead
    if (playheadMs !== null && playheadMs >= viewStart && playheadMs <= viewStart + viewSpan) {
      const x = xOf(playheadMs, w);
      ctx.fillStyle = "rgba(255,220,150,0.9)";
      ctx.fillRect(x, 0, 1.5, h);
      ctx.fillStyle = "rgba(255,220,150,0.12)";
      ctx.fillRect(x - 5, 0, 10, h);
    }
  }

  function drawHits(hits, laneIdx, laneY, laneH, w, asGhost) {
    const markW = Math.max(2.5, Math.min(6, (w - GUTTER_W) / (viewSpan / 60)));
    for (const hit of hits) {
      if (hit.ms < viewStart - 50 || hit.ms > viewStart + viewSpan + 50) continue;
      const i = laneIdx[hit.lane];
      if (i === undefined) continue;
      const x = xOf(hit.ms, w);
      const v = hit.velocity / 127;
      const hh = laneH * (0.22 + 0.66 * v);
      const y = laneY(i) + laneH - hh - 1;
      const color = LANE_COLORS[hit.lane] || "#e8b73c";
      if (asGhost) {
        ctx.globalAlpha = GHOST_ALPHA;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - markW / 2, y, markW, hh);
        ctx.globalAlpha = 1;
      } else {
        ctx.globalAlpha = 0.55 + 0.45 * v;
        ctx.fillStyle = color;
        ctx.fillRect(x - markW / 2, y, markW, hh);
        ctx.globalAlpha = 1;
      }
    }
  }

  // ---- minimap ---------------------------------------------------------------

  function drawMinimap() {
    if (!mmCtx) return;
    mmCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { w, h } = cssSize(mmCanvas);
    mmCtx.fillStyle = "#0b0e0c";
    mmCtx.fillRect(0, 0, w, h);
    if (!original) return;

    const d = data();
    const dur = durationMs();
    const laneN = d.lanes.length;
    const laneIdx = {};
    d.lanes.forEach((l, i) => { laneIdx[l] = i; });

    for (const hit of d.hits) {
      const x = (hit.ms / dur) * w;
      const y = 2 + (laneIdx[hit.lane] / laneN) * (h - 5);
      mmCtx.globalAlpha = 0.35 + 0.5 * (hit.velocity / 127);
      mmCtx.fillStyle = LANE_COLORS[hit.lane] || "#e8b73c";
      mmCtx.fillRect(x, y, 1.5, Math.max(1.5, (h - 5) / laneN - 1));
    }
    mmCtx.globalAlpha = 1;

    // viewport window
    const vx = (viewStart / dur) * w;
    const vw = Math.max(3, (viewSpan / dur) * w);
    mmCtx.strokeStyle = "rgba(255,220,150,0.8)";
    mmCtx.lineWidth = 1;
    mmCtx.strokeRect(vx + 0.5, 0.5, vw - 1, h - 1);
    mmCtx.fillStyle = "rgba(255,220,150,0.08)";
    mmCtx.fillRect(vx, 0, vw, h);

    // playhead
    if (playheadMs !== null) {
      mmCtx.fillStyle = "rgba(255,220,150,0.9)";
      mmCtx.fillRect((playheadMs / dur) * w, 0, 1, h);
    }
  }

  function redraw() { draw(); drawMinimap(); }

  // ---- interaction -----------------------------------------------------------

  function bindEvents() {
    let dragging = false;
    let dragStartX = 0;
    let dragStartView = 0;

    canvas.addEventListener("mousedown", (e) => {
      dragging = true;
      dragStartX = e.offsetX;
      dragStartView = viewStart;
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const { w } = cssSize(canvas);
      const dx = e.clientX - canvas.getBoundingClientRect().left - dragStartX;
      viewStart = dragStartView - dx * (viewSpan / (w - GUTTER_W));
      follow = false;
      redraw();
    });
    window.addEventListener("mouseup", () => { dragging = false; });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const { w } = cssSize(canvas);
      if (e.ctrlKey || e.metaKey) {
        const anchor = msAt(e.offsetX, w);
        const factor = Math.exp(e.deltaY * 0.002);
        viewSpan *= factor;
        clampView();
        viewStart = anchor - (e.offsetX - GUTTER_W) / (w - GUTTER_W) * viewSpan;
      } else {
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        viewStart += delta * (viewSpan / (w - GUTTER_W));
        follow = false;
      }
      redraw();
    }, { passive: false });

    canvas.addEventListener("dblclick", () => {
      viewStart = 0;
      viewSpan = durationMs();
      redraw();
    });

    let mmDragging = false;
    const mmJump = (e) => {
      const { w } = cssSize(mmCanvas);
      const ms = (e.offsetX / w) * durationMs();
      viewStart = ms - viewSpan / 2;
      follow = false;
      redraw();
    };
    mmCanvas.addEventListener("mousedown", (e) => { mmDragging = true; mmJump(e); });
    mmCanvas.addEventListener("mousemove", (e) => { if (mmDragging) mmJump(e); });
    window.addEventListener("mouseup", () => { mmDragging = false; });

    const ro = new ResizeObserver(() => {
      resizeCanvas(canvas);
      resizeCanvas(mmCanvas);
      redraw();
    });
    ro.observe(canvas.parentElement);
    ro.observe(mmCanvas.parentElement);
  }

  // ---- public API --------------------------------------------------------------

  function init(mainCanvas, minimapCanvas) {
    canvas = mainCanvas;
    ctx = canvas.getContext("2d");
    mmCanvas = minimapCanvas;
    mmCtx = mmCanvas.getContext("2d");
    resizeCanvas(canvas);
    resizeCanvas(mmCanvas);
    bindEvents();
    redraw();
  }

  function setData(orig, hum) {
    const isNewFile = !original || !orig || original.duration_ms !== orig.duration_ms;
    original = orig || null;
    humanised = hum || null;
    if (original && isNewFile) {
      // default window: first ~8 bars, or the whole song if shorter
      const bars = original.bars;
      viewStart = 0;
      viewSpan = bars.length > 8 ? bars[8].ms - bars[0].ms : durationMs();
    }
    redraw();
  }

  function setPlayhead(ms, followPlayhead = true) {
    playheadMs = ms;
    follow = followPlayhead && ms !== null;
    if (follow && (ms < viewStart || ms > viewStart + viewSpan * 0.8)) {
      viewStart = ms - viewSpan * 0.15;
    }
    redraw();
  }

  return { init, setData, setPlayhead, redraw };
})();

window.grooveScreen = grooveScreen;
