"""Build the loop video: a slowly drifting background plus the audio track.

The camera drift follows sine curves that complete whole cycles over the
loop, so the last frame flows straight back into the first.
"""

import subprocess

import numpy as np

SUPERSAMPLE = 2  # render the drift at 2x so sub-pixel motion stays smooth
OVERSCAN = 1.12  # how much bigger the background is than the frame


# Sky top, sky horizon, ridge near, ridge far, sun/moon colour (RGB 0-255).
PALETTES = {
    "dawn": [(40, 50, 100), (250, 180, 140), (30, 35, 55), (150, 130, 160), (255, 230, 190)],
    "forest": [(90, 150, 200), (210, 230, 220), (20, 50, 35), (120, 160, 150), (255, 250, 225)],
    "night": [(5, 8, 25), (30, 45, 85), (8, 10, 20), (45, 55, 90), (230, 235, 255)],
    "storm": [(60, 70, 85), (150, 160, 170), (25, 30, 35), (100, 110, 120), (200, 205, 210)],
    "sea": [(80, 140, 200), (240, 200, 170), (20, 50, 80), (110, 150, 180), (255, 240, 200)],
}


def render_scene(width, height, palette, seed):
    """Layered ridges under a gradient sky, as an (h, w, 3) uint8 array."""
    rng = np.random.default_rng(seed)
    top, horizon, near, far, sun = (np.array(c, dtype=float) for c in PALETTES[palette])
    y = np.linspace(0, 1, height)[:, None, None]
    x = np.linspace(0, 1, width)[None, :, None]
    img = top + (horizon - top) * y**1.5 * np.ones((1, width, 1))

    # Sun or moon with a soft glow.
    sx, sy = rng.uniform(0.3, 0.7), rng.uniform(0.25, 0.4)
    aspect = width / height
    d = np.sqrt(((x - sx) * aspect) ** 2 + (y - sy) ** 2)
    img += (sun - img) * np.clip(1 - d / 0.05, 0, 1) ** 0.3
    img += sun * 0.35 * np.exp(-d / 0.15)

    if palette == "night":
        stars = rng.random((height, width)) > 0.9993
        stars &= (y[:, :, 0] < 0.6)
        img[stars] = 255 * rng.uniform(0.5, 1.0, size=(stars.sum(), 1))

    # Ridges from far (pale) to near (dark), each a sum of random sines.
    layers = 5
    xs = np.linspace(0, 1, width)
    for i in range(layers):
        depth = i / (layers - 1)
        ridge = np.full(width, 0.42 + 0.3 * depth**1.3)
        for octave in range(5):
            freq = rng.uniform(0.8, 1.6) * 2**octave
            amp = (0.07 - 0.03 * depth) / (octave + 1) ** 1.8
            ridge += amp * np.sin(2 * np.pi * freq * xs + rng.uniform(0, 2 * np.pi))
        colour = far + (near - far) * depth
        mask = (y[:, :, 0] > ridge[None, :])[:, :, None]
        # Haze: each ridge fades slightly toward the sky near its crest.
        haze = np.clip((y - ridge[None, :, None]) * 6, 0, 1)
        shaded = colour * (0.75 + 0.25 * (1 - haze)) + (horizon - colour) * 0.15 * (1 - haze)
        img = np.where(mask, shaded, img)

    img += rng.normal(0, 1.2, img.shape)  # fine grain to avoid banding
    return np.clip(img, 0, 255).astype(np.uint8)


def write_ppm(path, rgb):
    h, w, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(f"P6 {w} {h} 255\n".encode())
        f.write(rgb.tobytes())


def build_video(background, audio_path, out_path, duration, width, height, fps, drift=0.9):
    """Encode `background` (still image) with a seamless drift + the audio."""
    sw, sh = width * SUPERSAMPLE, height * SUPERSAMPLE
    cw, ch = int(sw * OVERSCAN) // 2 * 2, int(sh * OVERSCAN) // 2 * 2
    T = duration
    vf = (
        f"scale={cw}:{ch}:force_original_aspect_ratio=increase,crop={cw}:{ch},"
        f"crop=w={sw}:h={sh}"
        f":x='(iw-ow)/2*(1+{drift}*sin(2*PI*t/{T}))'"
        f":y='(ih-oh)/2*(1+{drift}*sin(4*PI*t/{T}))',"
        f"scale={width}:{height}:flags=lanczos,format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
        "-loop", "1", "-framerate", str(fps), "-i", str(background),
        "-i", str(audio_path),
        "-vf", vf,
        "-t", f"{duration}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def repeat_video(src, out_path, times):
    """Loop a finished video `times` times without re-encoding."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-stream_loop", str(times - 1), "-i", str(src),
        "-c", "copy", "-movflags", "+faststart", str(out_path),
    ]
    subprocess.run(cmd, check=True)
