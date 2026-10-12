# Utube — ambient nature loop videos

Generates a **5-minute seamless loop video**: ambient music + nature sounds over
animated artwork. The main concept is the **Amazon rainforest**: an abstract
vector-art forest interior with light shafts through the canopy, swaying
leaves and vines, falling leaves, rain, drifting mist, blue morpho butterflies
and lightning flashing behind the trees, set to a jazzy lofi beat (swung
drums, Rhodes, bass, vinyl crackle) over a rainforest soundscape. Each
lightning strike is followed by its thunder about three seconds later.

The loop is seamless by design. The last frame and last audio sample flow
straight back into the first, so the video can be looped on YouTube, or
repeated into a 1-hour or 3-hour upload, without an audible or visible join.

## Requirements

- Python 3.10+
- `ffmpeg` (with libx264) on your `PATH`
- the Cairo graphics library (preinstalled on most Linux desktops; `brew install cairo` on macOS)

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 5-minute Amazon rainforest loop (the default preset)
python -m ambient_loop

# Preview a single frame (fast) before committing to a full render
python -m ambient_loop --still 42

# Quick low-res video preview
python -m ambient_loop --duration 30 --width 960 --height 540

# Also produce a 1-hour version (5 min x 12), no re-encoding
python -m ambient_loop --repeat 12

# A different variation (layout of plants, music phrases, sound events)
python -m ambient_loop --seed 7

# Older still-image presets: forest, rain, night, ocean, dawn
python -m ambient_loop --preset night
```

Output goes to `output/` (git-ignored). A full 5-minute 1080p Amazon render
takes roughly 15–25 minutes on a 4-core machine.

## Real field recordings (Freesound)

The Amazon preset layers real recordings over (or instead of) the synthesised
nature sounds:

| Layer | Freesound search | Replaces synth layers |
|---|---|---|
| `rainforest_ambience` | amazon rainforest ambience | insects, frogs, jungle birds |
| `rain_on_leaves` | rain on leaves jungle | jungle rain |

Thunder is always synthesised so it stays in sync with the on-screen lightning.

Two ways to supply them:

1. **Automatic:** create a free API key at <https://freesound.org/apiv2/apply>
   and set `FREESOUND_API_KEY`. Only **CC0 (public domain)** sounds are
   downloaded, so no attribution is required (a `credits.json` is kept anyway).
2. **Manual:** drop any audio files into `assets/sounds/<layer>/`,
   e.g. `assets/sounds/rain_on_leaves/my_rain.wav`.

Recordings are chained with crossfades and their tail is folded into their
head, so they loop seamlessly too. If none are available, the synthesised
layers are used. `--synth-only` forces that.

## How it works

| Piece | File | Notes |
|---|---|---|
| Animated rainforest | `ambient_loop/rainforest.py` | Vector shapes drawn with Cairo every frame, rendered in parallel and piped to ffmpeg |
| Lofi music | `ambient_loop/lofi.py` | Swung boom-bap drums, Rhodes ii–V–I–vi chords, bass, melody, sidechain pump, vinyl crackle |
| Other music styles | `ambient_loop/audio.py` | Pads + bells, or pads + kalimba + flute |
| Lightning timeline | `ambient_loop/storm.py` | Shared by the visuals (flash) and audio (thunder) |
| Synth nature sounds | `ambient_loop/audio.py` | Jungle rain with leaf drips, insects, tree frogs, tropical birds, distant thunder |
| Field recordings | `ambient_loop/recordings.py` | Freesound CC0 fetch + seamless looping |
| Still-image presets | `ambient_loop/video.py` | Generated landscape (or your `--image`) with a slow camera drift |
| Presets / CLI | `ambient_loop/cli.py` | Combines the above |

Seamless-loop tricks:
- The beat's tempo is nudged (76 → 76.8 BPM for 5 minutes) so a whole number
  of 8-bar sections fills the loop, and the groove runs straight through the join.
- Every animation completes a whole number of cycles per loop: wind gusts, leaf
  sway, light-shaft pulsing. Anything that travels (raindrops, falling leaves,
  mist, the butterfly) crosses the frame a whole number of times.
- Audio is rendered into a circular buffer: anything that runs past the end
  (a note's decay, a reverb tail, a thunder roll) wraps round to the start.
- Noise beds and filters work in the frequency domain, which is inherently periodic.
