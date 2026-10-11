# Utube — ambient nature loop videos

Generates a **5-minute seamless loop video**: ambient music + nature sounds over a
slowly drifting landscape. Everything is generated procedurally, so there are
no copyright/licensing issues with the audio or visuals.

The loop is seamless by design — the last frame and last audio sample flow
straight back into the first — so the video can be looped on YouTube, or
repeated into a 1-hour / 3-hour upload, without an audible or visible join.

## Requirements

- Python 3.10+
- `ffmpeg` (with libx264) on your `PATH`

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 5-minute forest loop (birds, stream, wind + major-key pads)
python -m ambient_loop

# Other presets: forest, rain, night, ocean, dawn
python -m ambient_loop --preset rain

# Different variation of the same preset
python -m ambient_loop --preset ocean --seed 7

# Your own background photo (it gets a slow, seamless camera drift)
python -m ambient_loop --preset forest --image my_photo.jpg

# Also produce a 1-hour version (5 min x 12), no re-encoding
python -m ambient_loop --preset night --repeat 12

# Quick preview while experimenting
python -m ambient_loop --duration 30 --width 640 --height 360
```

Output goes to `output/` (git-ignored): a `.wav` master, the `.mp4`, and the
generated background.

A full 5-minute 1080p render takes a few minutes on a laptop.

## How it works

| Piece | File | Notes |
|---|---|---|
| Music | `ambient_loop/audio.py` | Slow detuned chord pads cycling a 4-chord progression, sparse pentatonic bells, long reverb |
| Nature sounds | `ambient_loop/audio.py` | Rain, wind, stream, ocean, birds, crickets — synthesised from shaped noise and chirps |
| Visuals | `ambient_loop/video.py` | Layered ridges under a gradient sky, with a slow figure-of-eight camera drift |
| Presets / CLI | `ambient_loop/cli.py` | Combines the above |

Seamless-loop tricks:
- Audio is rendered into a circular buffer: anything that runs past the end
  (a bell's decay, a reverb tail, a bird call) wraps round to the start.
- Noise beds are shaped in the frequency domain, which is inherently periodic.
- Slow modulations (wind gusts, wave swells, camera drift) complete a whole
  number of cycles in the loop length.

## Ideas for next steps

- Drop in real field recordings (CC0 from freesound.org) as extra layers
- Animated elements: rain streaks, drifting fog, twinkling stars, water shimmer
- Use a real photo or AI-generated still as the background (`--image`)
- Title / thumbnail generation for the YouTube upload
