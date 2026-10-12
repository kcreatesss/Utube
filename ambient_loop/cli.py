import argparse
import time
from pathlib import Path

import numpy as np

from . import audio, rainforest, recordings, video

# Each preset: music settings (or None), synthesised nature layer gains, and
# either a still-background palette or an animated scene. "recordings" are
# real field recordings that, when available, replace the synthesised layers
# listed in "replaces".
PRESETS = {
    "amazon": {
        "music": {"key": "G", "mood": "major", "style": "rainforest", "chord_seconds": 24},
        "nature": {"jungle_rain": 0.45, "insects": 0.15, "frogs": 0.15,
                   "jungle_birds": 0.3, "thunder": 0.3},
        "recordings": {
            "rainforest_ambience": {"query": "amazon rainforest ambience", "gain": 0.6,
                                    "replaces": ["insects", "frogs", "jungle_birds"]},
            "rain_on_leaves": {"query": "rain on leaves jungle", "gain": 0.45,
                               "replaces": ["jungle_rain"]},
            "distant_thunder": {"query": "distant thunder rumble", "gain": 0.3,
                                "replaces": ["thunder"]},
        },
        "scene": "rainforest",
    },
    "forest": {
        "music": {"key": "D", "mood": "major"},
        "nature": {"birds": 0.5, "stream": 0.35, "wind": 0.25},
        "palette": "forest",
    },
    "rain": {
        "music": {"key": "A", "mood": "minor"},
        "nature": {"rain": 0.6, "wind": 0.2},
        "palette": "storm",
    },
    "night": {
        "music": {"key": "F", "mood": "minor", "bell_density": 0.1},
        "nature": {"crickets": 0.25, "wind": 0.3, "stream": 0.15},
        "palette": "night",
    },
    "ocean": {
        "music": {"key": "E", "mood": "major", "chord_seconds": 25},
        "nature": {"ocean": 0.7, "birds": 0.15, "wind": 0.15},
        "palette": "sea",
    },
    "dawn": {
        "music": {"key": "C", "mood": "major"},
        "nature": {"birds": 0.6, "wind": 0.2},
        "palette": "dawn",
    },
}


def main():
    p = argparse.ArgumentParser(
        prog="python -m ambient_loop",
        description="Generate a seamless ambient music + nature sounds loop video.",
    )
    p.add_argument("--preset", choices=PRESETS, default="amazon")
    p.add_argument("--duration", type=float, default=300, help="loop length in seconds (default 300)")
    p.add_argument("--seed", type=int, default=1, help="change for a different variation")
    p.add_argument("--image", type=Path, help="use your own background image instead of a generated one")
    p.add_argument("--no-music", action="store_true", help="nature sounds only")
    p.add_argument("--music-gain", type=float, default=0.6, help="music level relative to nature (default 0.6)")
    p.add_argument("--key", help="override the preset's musical key, e.g. C, F#, Bb is written A#")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--repeat", type=int, default=1, help="also write a version looped N times (e.g. 12 = 1 hour)")
    p.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    p.add_argument("--audio-only", action="store_true", help="only write the WAV")
    p.add_argument("--still", type=float, metavar="T", help="only render the frame at T seconds as a PNG")
    p.add_argument("--sounds-dir", type=Path, default=Path("assets/sounds"),
                   help="field recordings, one folder per layer (default assets/sounds)")
    p.add_argument("--synth-only", action="store_true", help="ignore field recordings")
    p.add_argument("--workers", type=int, help="frame-rendering processes (default: all cores)")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    music = None if args.no_music else dict(preset["music"])
    if music and args.key:
        music["key"] = args.key

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.preset}_{int(args.duration)}s_seed{args.seed}"
    wav_path = args.out / f"{stem}.wav"

    scene = preset.get("scene")
    if args.still is not None:
        if scene != "rainforest":
            p.error("--still needs a preset with an animated scene")
        png = args.out / f"{stem}_t{args.still:g}.png"
        rainforest.save_still(png, args.width, args.height, args.duration, args.still, args.seed)
        print(f"  -> {png}")
        return

    t0 = time.time()
    print(f"Rendering audio ({args.duration:.0f}s, preset '{args.preset}')...")
    nature = dict(preset["nature"])
    loops = []
    if not args.synth_only:
        n = int(round(args.duration * audio.SAMPLE_RATE))
        rng = np.random.default_rng(args.seed)
        for name, rec in preset.get("recordings", {}).items():
            loop = recordings.load_layer(name, rec["query"], n, args.sounds_dir, rng)
            if loop is None:
                continue
            print(f"  using field recordings for '{name}'")
            loops.append((loop, rec["gain"]))
            for replaced in rec["replaces"]:
                nature.pop(replaced, None)
    mix = audio.render_mix(args.duration, args.seed, music, nature, args.music_gain, loops)
    audio.write_wav(wav_path, mix)
    print(f"  -> {wav_path} ({time.time() - t0:.1f}s)")
    if args.audio_only:
        return

    if scene == "rainforest" and args.image is None:
        mp4_path = args.out / f"{stem}.mp4"
        print("Rendering animated scene...")
        rainforest.build_video(wav_path, mp4_path, args.duration, args.width, args.height,
                               args.fps, args.seed, args.workers)
        print(f"  -> {mp4_path}")
        finish(args, mp4_path, stem, t0)
        return

    background = args.image
    if background is None:
        background = args.out / f"{stem}_bg.ppm"
        print("Rendering background...")
        bg = video.render_scene(
            int(args.width * video.SUPERSAMPLE * video.OVERSCAN),
            int(args.height * video.SUPERSAMPLE * video.OVERSCAN),
            preset.get("palette", "forest"),
            args.seed,
        )
        video.write_ppm(background, bg)

    mp4_path = args.out / f"{stem}.mp4"
    print("Encoding video...")
    video.build_video(background, wav_path, mp4_path, args.duration, args.width, args.height, args.fps)
    print(f"  -> {mp4_path}")
    finish(args, mp4_path, stem, t0)


def finish(args, mp4_path, stem, t0):
    if args.repeat > 1:
        long_path = args.out / f"{stem}_x{args.repeat}.mp4"
        video.repeat_video(mp4_path, long_path, args.repeat)
        print(f"  -> {long_path}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
