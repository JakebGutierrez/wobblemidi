# Demo files

Before/after MIDI pairs, generated from the shipped rock profile with pinned
seeds — every file here is reproducible byte-for-byte:

```bash
python scripts/make_demo_pack.py    # the before/after pairs (seed 42, engine defaults)
python scripts/make_demo.py         # the phi A/B trio (seed 7, intensity 0.85)
```

## What each pair demonstrates

| Files | Pattern | What to listen for |
|---|---|---|
| `rock_ghosts_input.mid` → `rock_ghosts_humanised.mid` | 8 bars, ~95 BPM: snare ghosts + backbeats, busy three-level 16th hi-hats | **The flagship pair.** Ghost notes stay ghosts and backbeats stay solid (relative velocity tiering); the hi-hat accent contour survives with human-scale variation instead of machine-gunning; timing breathes as one pocket rather than per-hit jitter. |
| `four_floor_input.mid` → `four_floor_humanised.mid` | 8 bars, ~95 BPM: four-on-the-floor, backbeat snare, straight 8th hats | The plainest groove: the programmed rigidity is gone at the default intensity, but nothing sounds loose or drifts audibly off the grid. |
| `flam_beat_input.mid` → `flam_beat_humanised.mid` (+ `flam_beat_uncoupled.mid`) | 4 bars, 120 BPM: rock beat with snare flams on beat 4 of bars 2 & 4 | Flam preservation (the coupling window): in `humanised` the grace→main spacing is preserved exactly while the flam moves with the groove. `uncoupled` is the same seed with `--groove-tightness 0` — one flam audibly collapses into a single thick hit. |
| `rock_4bar_input.mid` → `rock_4bar_phi0.mid` / `rock_4bar_phi05.mid` | 4 bars, 120 BPM, intensity 0.85 (deliberately high) | The groove-clock A/B: same seed, same amount of timing spread. `phi0` times every hit independently (twitchy); `phi05` shares one drifting clock (pocket). |

All `_input` files are programmed dead on the grid with a flat velocity
palette — that is the point of the demo.

## Rendering checklist (Logic + Superior Drummer 3)

Renders live in `demo/audio/`, named `<pattern>_before.mp3` / `<pattern>_after.mp3`
(the main README links these names — keep them exact).

**Required (linked from the main README):**

- [ ] `rock_ghosts_before.mp3` ← `rock_ghosts_input.mid`
- [ ] `rock_ghosts_after.mp3` ← `rock_ghosts_humanised.mid`
- [ ] `four_floor_before.mp3` ← `four_floor_input.mid`
- [ ] `four_floor_after.mp3` ← `four_floor_humanised.mid`

**Optional (nice to have):**

- [ ] `flam_beat_after.mp3` ← `flam_beat_humanised.mid`
- [ ] `flam_beat_uncoupled.mp3` ← `flam_beat_uncoupled.mid`

Render settings that matter:

1. **Same kit, same mix, same level for both files of a pair** — the comparison
   is the MIDI, nothing else may differ.
2. Any acoustic **rock kit** preset works; pick a snare where ghost notes at
   velocity ~26–30 are clearly audible but clearly quiet (the rock_ghosts pair
   lives or dies on this). Avoid heavy room compression — it eats the timing
   and dynamic detail the demo exists to show.
3. **Disable SD3's own humanise/velocity randomisation**, and do not quantise
   or apply Logic's MIDI transforms — the files must pass through untouched.
4. Import the file tempo when Logic asks (rock_ghosts/four_floor are ~95 BPM,
   flam_beat/rock_4bar are 120 BPM), or set the project tempo to match.
5. Export as MP3 (or M4A) — small enough to commit, and GitHub serves them
   playable in the browser. 256 kbps is plenty.

## GUI capture plan (optional, for the main README)

One short GIF (~15 s, `demo/gui_workflow.gif`), captured at 2× speed:

1. Launch `wobblemidi-gui`, drag `rock_ghosts_input.mid` onto the window.
2. Hit HUMANISE with defaults; the note display updates.
3. Nudge INTENSITY up, HUMANISE again; click a lane to show lane-scoped intensity.
4. Export via the save dialog.

macOS: Cmd+Shift+5 screen recording → convert with
`ffmpeg -i in.mov -vf "fps=12,scale=880:-1" demo/gui_workflow.gif`.
