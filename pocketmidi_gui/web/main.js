// main.js — top-level UI wiring. Phase 0: power LED lights when the Python
// backend answers over the bridge.

(async () => {
  try {
    const answer = await bridge.ping();
    if (answer === "pong") {
      document.getElementById("power-led").classList.add("on");
    }
  } catch (err) {
    console.error("backend bridge failed:", err);
  }
})();

function toast(message, ms = 3200) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), ms);
}

window.toast = toast;
