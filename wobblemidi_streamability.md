# wobblemidi — streamability map (plugin-readiness classification)

_Written 2026-07-06 (Fable window, Session 3). Classifies every engine behaviour by what
it would take to run as a **real-time streaming MIDI FX**, versus the shipped offline
whole-clip model. The endgame plugin (JUCE/C++/AU, see `wobblemidi_porting_contract.md`)
is **deliberately NOT a streaming FX** — it is an SD3-style drag-in / process / drag-out
offline MIDI environment operating on whole clips. This map is the evidence for that
decision: it shows which behaviours the offline model gets for free and what a
hypothetical realtime mode would cost. Line references: engine at the commit introducing
this file._

## Classes

- **streamable** — computable per event from already-seen events plus static data
  (profile, params, host tempo/meter). No lookahead.
- **streamable-with-latency (bound)** — computable with a bounded lookahead/delay; the
  bound is stated. In plugin terms: plugin-delay-compensation of at least that bound.
- **offline-only** — requires the whole clip (or an unbounded lookahead) to reproduce the
  reference behaviour exactly. Streaming approximations exist for some, but they would
  produce *different output* than the reference engine and fail the golden/harness
  equivalence that defines correctness.

In the **offline-clip plugin model, every row below is available by design** — the clip
is fully in hand before processing. The classification below is therefore about a
hypothetical realtime mode, not a porting blocker.

## Context acquisition (standalone-tool work a host does for free)

| Behaviour | Class (as coded) | Notes |
|---|---|---|
| Tempo-map construction (`build_tempo_map`, `midi_utils.py:45-61`) | offline-only — whole-file scan | In a plugin the host supplies the tempo map for the clip region; this pass exists because the standalone tool has no host. Reclassifies to "host-provided" in-plugin. |
| Meter detection + 6/8-mix rejection (`detect_meter`, `midi_utils.py:172-221`; `is_four_four`, `:158-169`) | offline-only — whole-file scan | Same: host time-signature substitutes. The 6/8-vs-16th grid choice (`humanise.py:465-472`) then follows per clip. |
| Drum-channel filter (`DRUM_CHANNEL`, `humanise.py:33`, `will_shift` at `:546-552`) | streamable | Pure per-message predicate (channel, note number, bucket availability). |

## Whole-clip statistics (the genuinely offline core)

| Behaviour | Class | Notes |
|---|---|---|
| **Relative velocity tiering** (B4: `vels_by_group` scan `humanise.py:481-496`, `_file_tier_thresholds` `:123-186`) | **offline-only, fundamentally** | Tier thresholds are percentiles / gap analysis over the **whole clip's** per-instrument velocities, and the tier selects both the velocity *and timing* bucket for every hit (`_lookup` thresholds override, `:205-212`). A running-percentile approximation would change hit outcomes and diverge from the reference. This is the flagship reason the plugin is offline-clip. |
| Per-tick min-eff for chords (`tick_min_eff`, `humanise.py:510,563-566`) | streamable-with-latency (~0: same-tick barrier across lanes) | Needs all tracks' shiftable hits at one original tick before placing the anchor — a cross-lane synchronisation point, trivial offline, a merge barrier when streaming. |

## Per-hit pipeline

| Behaviour | Class | Notes |
|---|---|---|
| Grid quantise (`quantise_to_grid`, `midi_utils.py:125-144`) + bar position (`grid_position_in_bar`, `:147-155`) | streamable | Pure functions of tick, PPQ, grid — given host meter/bar alignment. |
| Bucket lookup + fallback chain (`_lookup`, `humanise.py:189-246`) | streamable *given tier* | Static profile data; the offline dependency is the tier input (above). |
| KDE sampling (`_sample_bucket`, `humanise.py:303-321`) | streamable | Per-hit RNG draw from static per-bucket KDEs. |
| De-bias / lean arithmetic (`mu_center`, `mu_debias`, `lean`; `humanise.py:883-925`) | streamable | Per-bucket constants from the profile `_meta`. |
| AR(1) timing clock (`GrooveDrift.step`, `humanise.py:375-383`, stepped at `:710,:919`) | streamable | Causal recursion on past state only. Kit-wide: consumes hits in absolute-time order across lanes — natural in a stream, but reproducibility then depends on a deterministic cross-lane merge order (guaranteed offline by the stable tick sort, `humanise.py:612`). |
| Kick velocity clock (`PHI_VEL`, `_new_velocity`, `humanise.py:512-524`) | streamable | Causal, kick-only. |
| Velocity application (round + clamp, `humanise.py:524`) | streamable | Per-hit, given tier. |
| **Negative timing offsets** (early hits; sampled offset < 0 applied at `:891,:923`) | **streamable-with-latency** | An early hit must be emitted before its notated time. Measured bound from the shipped profile (samples are clamped to each bucket's retained range, `:316`): the most-early bucket minimum is **−120 ms**, so the sample term needs ≤ 120 ms × eff lookahead at full intensity (≈ 42 ms at the 0.35 default); drift adds up to ≈ √((1+φ)/(1−φ)) × the clamped centred sample (≈ 1.5× at φ=0.4) and the gaussian residual is tail-unbounded — a practical PDC budget is **~150 ms × eff**. |
| Same-tick chord coupling (`chord_tick` / `chord_anchor_abs`, `humanise.py:894-909,:926,:956-959`) | streamable-with-latency (~0 + the anchor's own latency) | A coupled member targets the anchor's *emitted* tick — available once the anchor (same original tick) is placed. Cross-track members require the same-tick merge barrier. |
| **12 ms coupling window** (cluster scan `humanise.py:636-666`, `_plan_cluster` `:668-807`) | streamable-with-latency (12 ms + cluster span) **for membership**; see next row for the clamp | Membership needs every shiftable hit within 12 ms of the cluster's first hit before the shared delta is fixed. |
| Forward-window clamping: `next_fixed` (`humanise.py:569-575`), paired note-off ceilings (`:577-592`, used `:939-943`), cluster walls (`_next_wall`, `:730-739`) | **offline-only as coded** — the scans for the next fixed event / paired off are unbounded lookahead | Could be reformulated with a bounded horizon H ≈ the max forward shift (same ~150 ms × eff budget as the early side, plus 12 ms window): "streamable-with-latency under redesign". But that is a *redesign* — corner-case behaviour (a distant wall inside a long window; a long note's off) would differ from the reference engine. |
| Elastic member note-offs (`humanise.py:963-970`) | streamable | Rides behind its already-placed note-on; past state only. |
| Minimum-separation / hold fallbacks (`EPSILON_TICKS` lower bounds `:933-938`, hold `:945-948`, cluster hold `:784-801`) | streamable | Causal (previous emitted state), once the ceiling inputs above are available. |
| Delta-time re-encode (pass 3, `humanise.py:978-987`) | streamable-with-latency (= max early shift) | Events can only be flushed once no earlier-tick event can still arrive — tied to the negative-offset budget. |

## Cross-cutting: seed semantics

Per-clip determinism (`wobblemidi_determinism.md`) is a property of processing a fixed
event list in a fixed order. In a live streaming FX, "same seed" has no stable meaning
across passes (event arrival defines the draw order). The offline-clip model preserves
the reproducibility contract — including in the plugin.

## Conclusion (the plugin-readiness verdict)

The **offline whole-clip model needs no redesign anywhere**: every offline-only row is
trivially satisfied when the clip is in hand, host tempo/meter replace the two
context-acquisition scans, and the engine's three-pass structure ports 1:1. This is
settled decision 3 confirmed by inventory, not by taste.

A hypothetical realtime streaming mode (explicitly **not planned**) would cost:
- relative velocity tiering (B4) — dropped or approximated with different output;
- exact forward-window semantics — replaced by a bounded-horizon approximation;
- PDC latency ≈ 150 ms × eff (+12 ms coupling window) to honour early hits and clamps;
- cross-lane merge barriers at every shared tick;
- the per-clip seed-reproducibility contract.

Any such mode would fail byte-equivalence with the reference engine by construction and
would need its own Tier 2 (distributional) validation — see the two-tier correctness
definition in `wobblemidi_porting_contract.md`.
