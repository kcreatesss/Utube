"""Real field recordings: fetch CC0 sounds from Freesound and loop them seamlessly.

Sounds are cached under `assets/sounds/<layer>/`. You can also drop your own
recordings (any format ffmpeg reads) into those folders by hand; they are
used in preference to downloading.

Downloading needs a free Freesound API key in the FREESOUND_API_KEY
environment variable (https://freesound.org/apiv2/apply). Only sounds
licensed CC0 (public domain) are fetched, so no attribution is required,
but a credits.json is written next to each download anyway.
"""

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

API = "https://freesound.org/apiv2/search/text/"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aiff", ".aif", ".opus"}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def search(query, api_key, min_seconds=60, max_seconds=1200, limit=5):
    params = {
        "query": query,
        "filter": f'license:"Creative Commons 0" duration:[{min_seconds} TO {max_seconds}]',
        "fields": "id,name,username,license,duration,previews,url",
        "sort": "rating_desc",
        "page_size": limit,
        "token": api_key,
    }
    with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=30) as r:
        return json.load(r)["results"]


def fetch(query, dest, api_key, count=1):
    """Download the top `count` CC0 results for `query` into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    credits_path = dest / "credits.json"
    credits = json.loads(credits_path.read_text()) if credits_path.exists() else []
    for result in search(query, api_key)[:count]:
        path = dest / f"{result['id']}.ogg"
        if not path.exists():
            print(f"  downloading freesound #{result['id']} '{result['name']}' by {result['username']}")
            urllib.request.urlretrieve(result["previews"]["preview-hq-ogg"], path)
            credits.append({k: result[k] for k in ("id", "name", "username", "license", "url")})
    credits_path.write_text(json.dumps(credits, indent=2))


def decode(path):
    """Any audio file -> float32 stereo at SAMPLE_RATE, via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SAMPLE_RATE), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def _crossfade_curves(fade):
    t = np.linspace(0, 1, fade)[:, None]
    return np.sin(t * np.pi / 2), np.cos(t * np.pi / 2)  # equal power


def crossfade_concat(a, b, fade):
    fade_in, fade_out = _crossfade_curves(fade)
    return np.concatenate([a[:-fade], a[-fade:] * fade_out + b[:fade] * fade_in, b[fade:]])


def seamless_loop(clips, n, fade_seconds=4.0):
    """Chain clips with crossfades until long enough, then fold the tail into the head.

    The result's last sample flows into its first: out[n-1] is clip[n-1]
    and out[0] is (almost entirely) clip[n], the sample that followed it.
    """
    fade = int(fade_seconds * SAMPLE_RATE)
    clips = [c for c in clips if len(c) > 2 * fade]
    if not clips:
        raise ValueError("recordings are too short to loop")
    x = clips[0]
    i = 1
    while len(x) < n + fade:
        x = crossfade_concat(x, clips[i % len(clips)], fade)
        i += 1
    fade_in, fade_out = _crossfade_curves(fade)
    out = x[:n].copy()
    out[:fade] = x[:fade] * fade_in + x[n:n + fade] * fade_out
    return out


def load_layer(name, query, n, sounds_dir, rng):
    """Return a looped recording for this layer, or None if none is available."""
    folder = sounds_dir / slug(name)
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTENSIONS) if folder.exists() else []
    if not files:
        api_key = os.environ.get("FREESOUND_API_KEY")
        if not api_key:
            return None
        try:
            fetch(query, folder, api_key)
        except OSError as e:
            print(f"  could not reach Freesound for '{name}': {e}")
            return None
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    clips = [decode(p) for p in files]
    order = rng.permutation(len(clips))
    return seamless_loop([clips[i] for i in order], n)
