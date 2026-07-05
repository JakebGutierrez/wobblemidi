// knobs.js — rotary knob + toggle components.
//
// Input gesture is vertical drag (the DAW-plugin standard): drag up/down over
// the knob, shift-drag for fine control, scroll wheel to nudge, double-click
// to reset to default. The knob face stays fixed; only the pointer rotor
// rotates (so the specular highlight doesn't spin with it).

const ANGLE_MIN = -135;
const ANGLE_MAX = 135;
const DRAG_RANGE_PX = 180;   // full-range vertical drag distance
const FINE_FACTOR = 0.1;     // shift = 10x finer

function makeKnob({ mount, min, max, value, def, fmt, onInput, onReset }) {
  const knobEl = mount.querySelector(".knob");
  const rotor = mount.querySelector(".knob-rotor");
  const valueEl = mount.querySelector(".ctl-value");
  fmt = fmt || ((v) => v.toFixed(2));
  let v = value;

  function angle() {
    return ANGLE_MIN + ((v - min) / (max - min)) * (ANGLE_MAX - ANGLE_MIN);
  }

  function render() {
    rotor.style.transform = `rotate(${angle()}deg)`;
    valueEl.textContent = fmt(v);
  }

  function set(nv, fire = true) {
    v = Math.min(max, Math.max(min, nv));
    render();
    if (fire && onInput) onInput(v);
  }

  let dragging = false;
  let startY = 0;
  let startV = 0;

  knobEl.addEventListener("mousedown", (e) => {
    dragging = true;
    startY = e.clientY;
    startV = v;
    document.body.style.cursor = "ns-resize";
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const fine = e.shiftKey ? FINE_FACTOR : 1;
    const dv = ((startY - e.clientY) / DRAG_RANGE_PX) * (max - min) * fine;
    set(startV + dv);
  });
  window.addEventListener("mouseup", () => {
    if (dragging) {
      dragging = false;
      document.body.style.cursor = "";
    }
  });

  knobEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    const fine = e.shiftKey ? FINE_FACTOR : 1;
    set(v - (e.deltaY / 600) * (max - min) * fine);
  }, { passive: false });

  // double-click resets to default unless the owner overrides it (the GUI uses
  // this for lane scope: reset = "clear this lane's override", not "go to 0.35")
  knobEl.addEventListener("dblclick", () => {
    if (onReset) onReset();
    else set(def);
  });

  render();
  return { get: () => v, set };
}

function makeSlider({ mount, min, max, value, def, detent, fmt, onInput }) {
  // Horizontal slider (LEAN): drag along the track, wheel nudges, shift = fine,
  // double-click = default. `detent` snaps values within its radius to `def`.
  const track = mount.querySelector(".slider-track");
  const thumb = mount.querySelector(".slider-thumb");
  const valueEl = mount.querySelector(".ctl-value");
  fmt = fmt || ((v) => v.toFixed(2));
  let v = value;

  function render() {
    const frac = (v - min) / (max - min);
    thumb.style.left = `${frac * 100}%`;
    valueEl.textContent = fmt(v);
    mount.classList.toggle("at-default", v === def);
  }

  function set(nv, fire = true) {
    nv = Math.min(max, Math.max(min, nv));
    if (detent && Math.abs(nv - def) < detent) nv = def;
    v = nv;
    render();
    if (fire && onInput) onInput(v);
  }

  function valueAtPointer(e) {
    const r = track.getBoundingClientRect();
    return min + ((e.clientX - r.left) / r.width) * (max - min);
  }

  let dragging = false;
  track.addEventListener("mousedown", (e) => {
    dragging = true;
    set(valueAtPointer(e));
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    if (e.shiftKey) {
      const r = track.getBoundingClientRect();
      set(v + (e.movementX / r.width) * (max - min) * FINE_FACTOR);
    } else {
      set(valueAtPointer(e));
    }
  });
  window.addEventListener("mouseup", () => { dragging = false; });

  mount.addEventListener("wheel", (e) => {
    e.preventDefault();
    const fine = e.shiftKey ? FINE_FACTOR : 1;
    set(v - (e.deltaY / 600) * (max - min) * fine);
  }, { passive: false });

  mount.addEventListener("dblclick", () => set(def));

  render();
  return { get: () => v, set };
}

function makeToggle({ mount, value, labels, onInput }) {
  const toggleEl = mount.querySelector(".toggle");
  const valueEl = mount.querySelector(".ctl-value");
  labels = labels || ["OFF", "ON"];
  let v = !!value;

  function render() {
    toggleEl.classList.toggle("on", v);
    valueEl.textContent = labels[v ? 1 : 0];
  }

  function set(nv, fire = true) {
    v = !!nv;
    render();
    if (fire && onInput) onInput(v);
  }

  toggleEl.addEventListener("click", () => set(!v));

  render();
  return { get: () => v, set };
}

window.makeKnob = makeKnob;
window.makeToggle = makeToggle;
window.makeSlider = makeSlider;
