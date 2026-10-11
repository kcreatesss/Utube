"""Procedural ambient music and nature sounds that loop seamlessly.

Every layer is rendered into a circular buffer exactly `duration` seconds
long: events that run past the end wrap around to the start, noise is shaped
in the frequency domain (which is inherently periodic), and slow modulations
complete a whole number of cycles. The result has no seam when looped.
"""

import wave

import numpy as np

SAMPLE_RATE = 44100

# Semitone offsets from the root for a few moods.
SCALES = {
    "major": [0, 2, 4, 7, 9],  # major pentatonic
    "minor": [0, 3, 5, 7, 10],  # minor pentatonic
}

# Chord progressions as scale degrees (semitones from root) per chord.
PROGRESSIONS = {
    "major": [[0, 4, 7, 11], [5, 9, 12, 16], [9, 12, 16, 19], [7, 11, 14, 17]],
    "minor": [[0, 3, 7, 10], [8, 12, 15, 19], [3, 7, 10, 14], [5, 8, 12, 15]],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def root_midi(key):
    """'D' -> MIDI note for D3, a comfortable pad register."""
    return 48 + NOTE_NAMES.index(key.upper())


# ---------------------------------------------------------------------------
# Circular-buffer helpers
# ---------------------------------------------------------------------------

def add_wrapped(buf, start, signal):
    """Add `signal` into `buf` at `start`, wrapping past the end to the start."""
    n = len(buf)
    start %= n
    end = start + len(signal)
    if end <= n:
        buf[start:end] += signal
    else:
        split = n - start
        buf[start:] += signal[:split]
        add_wrapped(buf, 0, signal[split:])


def shaped_noise(n, rng, low_hz, high_hz, tilt=0.0):
    """Periodic noise band-limited to [low_hz, high_hz], normalised to unit RMS.

    `tilt` is the spectral slope in dB/octave (e.g. -3 for pink-ish).
    """
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    f = np.maximum(freqs, 1.0)
    # Smooth (2nd order-ish) band edges rather than brick walls.
    mask = 1 / (1 + (low_hz / f) ** 4) / (1 + (f / high_hz) ** 4)
    if tilt:
        mask *= (f / 1000.0) ** (tilt / 6.02)
    out = np.fft.irfft(spectrum * mask, n)
    return out / (np.sqrt(np.mean(out**2)) + 1e-12)


def slow_envelope(n, rng, max_hz, floor=0.0):
    """Periodic random envelope in [floor, 1] varying no faster than max_hz."""
    # Build at a low control rate, then interpolate up (cheap for long buffers).
    ctrl_rate = max(20.0, max_hz * 8)
    m = max(16, int(n / SAMPLE_RATE * ctrl_rate))
    spectrum = np.fft.rfft(rng.standard_normal(m))
    freqs = np.fft.rfftfreq(m, 1 / ctrl_rate)
    spectrum[freqs > max_hz] = 0
    spectrum[0] = 0
    ctrl = np.fft.irfft(spectrum, m)
    ctrl = (ctrl - ctrl.min()) / (np.ptp(ctrl) + 1e-12)
    # Interpolate over a closed loop so the end meets the start.
    x = np.linspace(0, m, n, endpoint=False)
    env = np.interp(x, np.arange(m + 1), np.append(ctrl, ctrl[0]))
    return floor + (1 - floor) * env


def circular_reverb(stereo, rng, seconds=3.0, wet=0.35):
    """Convolve with a decaying-noise impulse response, circularly (seamless)."""
    n = len(stereo)
    ir_len = int(seconds * SAMPLE_RATE)
    t = np.arange(ir_len) / SAMPLE_RATE
    out = np.empty_like(stereo)
    for ch in range(2):
        ir = np.zeros(n)
        ir[:ir_len] = rng.standard_normal(ir_len) * np.exp(-t * 6.9 / seconds)
        ir /= np.sqrt(np.sum(ir**2))
        wet_sig = np.fft.irfft(np.fft.rfft(stereo[:, ch]) * np.fft.rfft(ir), n)
        out[:, ch] = (1 - wet) * stereo[:, ch] + wet * wet_sig
    return out


def pan(mono, position):
    """Equal-power pan; position in [-1 (left), 1 (right)]."""
    angle = (position + 1) * np.pi / 4
    return np.stack([mono * np.cos(angle), mono * np.sin(angle)], axis=1)


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------

def render_pads(n, rng, key, mood, chord_seconds):
    """Slow, overlapping chord pads following a looping progression."""
    out = np.zeros((n, 2))
    duration = n / SAMPLE_RATE
    progression = PROGRESSIONS[mood]
    # Fit a whole number of chords into the loop.
    num_chords = max(len(progression), round(duration / chord_seconds))
    chord_len = n // num_chords
    voice_len = int(chord_len * 1.6)  # overlap neighbours for smooth crossfades
    t = np.arange(voice_len) / SAMPLE_RATE
    fade = np.sin(np.linspace(0, np.pi, voice_len)) ** 2
    root = root_midi(key)

    for i in range(num_chords):
        chord = progression[i % len(progression)]
        voice = np.zeros((voice_len, 2))
        for j, degree in enumerate(chord):
            f = midi_to_hz(root + degree)
            # Detuned partials give a soft, chorused pad.
            for detune, amp in ((-0.12, 1.0), (0.12, 1.0), (0.0, 0.25)):
                ff = f * 2 ** (detune / 12)
                tone = np.sin(2 * np.pi * ff * t + rng.uniform(0, 2 * np.pi))
                tone += 0.2 * np.sin(2 * np.pi * 2 * ff * t)
                voice += pan(amp * tone, (j / max(1, len(chord) - 1)) * 1.2 - 0.6 + detune)
        # Sub-bass root an octave down.
        voice += pan(0.6 * np.sin(2 * np.pi * midi_to_hz(root + chord[0] - 12) * t), 0)
        start = i * chord_len - (voice_len - chord_len) // 2
        add_wrapped(out[:, 0], start, voice[:, 0] * fade)
        add_wrapped(out[:, 1], start, voice[:, 1] * fade)
    return out


def render_bells(n, rng, key, mood, density):
    """Sparse, soft pentatonic bell notes."""
    out = np.zeros((n, 2))
    scale = SCALES[mood]
    root = root_midi(key) + 24
    note_len = int(5 * SAMPLE_RATE)
    t = np.arange(note_len) / SAMPLE_RATE
    env = (1 - np.exp(-t * 60)) * np.exp(-t * 1.1)
    pos = 0
    while pos < n:
        pos += int(rng.uniform(0.5, 1.5) * SAMPLE_RATE / density)
        if pos >= n:
            break
        note = root + rng.choice(scale) + 12 * rng.integers(0, 2)
        f = midi_to_hz(note)
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t * 3)
        stereo = pan(tone * env * rng.uniform(0.4, 1.0), rng.uniform(-0.7, 0.7))
        add_wrapped(out[:, 0], pos, stereo[:, 0])
        add_wrapped(out[:, 1], pos, stereo[:, 1])
    return out


def render_music(n, rng, key="D", mood="major", chord_seconds=20.0, bell_density=0.15):
    pads = render_pads(n, rng, key, mood, chord_seconds)
    bells = render_bells(n, rng, key, mood, bell_density)
    mix = normalize_rms(pads) + 0.5 * normalize_rms(bells)
    return circular_reverb(mix, rng, seconds=4.0, wet=0.45)


# ---------------------------------------------------------------------------
# Nature layers (each returns an (n, 2) array at roughly unit RMS)
# ---------------------------------------------------------------------------

def rain(n, rng):
    bed = np.stack([shaped_noise(n, rng, 500, 9000, tilt=-3) for _ in range(2)], axis=1)
    drops = np.zeros((n, 2))
    drop_len = int(0.03 * SAMPLE_RATE)
    drop_env = np.exp(-np.arange(drop_len) / SAMPLE_RATE * 180)
    for pos in rng.integers(0, n, size=int(n / SAMPLE_RATE * 25)):
        d = np.diff(rng.standard_normal(drop_len + 1)) * drop_env * rng.uniform(0.2, 1)
        stereo = pan(d, rng.uniform(-1, 1))
        add_wrapped(drops[:, 0], pos, stereo[:, 0])
        add_wrapped(drops[:, 1], pos, stereo[:, 1])
    return bed + 0.5 * normalize_rms(drops)


def wind(n, rng):
    out = np.zeros((n, 2))
    for ch in range(2):
        gust = slow_envelope(n, rng, max_hz=0.08, floor=0.15)
        out[:, ch] = shaped_noise(n, rng, 60, 900, tilt=-4) * gust
    return out


def stream(n, rng):
    out = np.zeros((n, 2))
    for ch in range(2):
        burble = slow_envelope(n, rng, max_hz=6, floor=0.4)
        out[:, ch] = shaped_noise(n, rng, 300, 5000, tilt=-2) * burble
    return out


def ocean(n, rng):
    duration = n / SAMPLE_RATE
    waves = max(1, round(duration / 10))  # one swell every ~10 s, whole cycles
    phase = np.arange(n) / n * waves * 2 * np.pi
    swell = (0.5 - 0.5 * np.cos(phase)) ** 2
    variation = slow_envelope(n, rng, max_hz=0.03, floor=0.5)
    env = 0.15 + 0.85 * swell * variation
    out = np.zeros((n, 2))
    for ch in range(2):
        out[:, ch] = shaped_noise(n, rng, 80, 4000, tilt=-3) * env
    return out


def birds(n, rng, calls_per_minute=10):
    out = np.zeros((n, 2))
    num_calls = int(n / SAMPLE_RATE / 60 * calls_per_minute)
    for pos in rng.integers(0, n, size=num_calls):
        base = rng.uniform(2500, 5000)
        position = rng.uniform(-0.9, 0.9)
        level = rng.uniform(0.3, 1.0)
        for s in range(rng.integers(2, 6)):
            syl_len = int(rng.uniform(0.05, 0.15) * SAMPLE_RATE)
            t = np.arange(syl_len) / SAMPLE_RATE
            sweep = base * (1 + rng.uniform(-0.3, 0.4) * t / t[-1])
            phase = 2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE
            env = np.sin(np.pi * np.arange(syl_len) / syl_len) ** 2
            chirp = pan(np.sin(phase) * env * level, position)
            start = pos + s * int(rng.uniform(0.1, 0.2) * SAMPLE_RATE)
            add_wrapped(out[:, 0], start, chirp[:, 0])
            add_wrapped(out[:, 1], start, chirp[:, 1])
    return circular_reverb(out, rng, seconds=1.5, wet=0.3)


def crickets(n, rng, num_crickets=4):
    out = np.zeros((n, 2))
    for _ in range(num_crickets):
        carrier_hz = rng.uniform(4000, 5000)
        position = rng.uniform(-0.8, 0.8)
        pulse_len = int(0.018 * SAMPLE_RATE)
        t = np.arange(pulse_len) / SAMPLE_RATE
        pulse = np.sin(2 * np.pi * carrier_hz * t) * np.sin(np.pi * t / t[-1]) ** 2
        period = rng.uniform(0.5, 1.0)
        pos = int(rng.uniform(0, period) * SAMPLE_RATE)
        while pos < n:
            for k in range(3):  # three pulses per chirp
                p = pan(pulse * rng.uniform(0.6, 1.0), position)
                add_wrapped(out[:, 0], pos + k * int(0.035 * SAMPLE_RATE), p[:, 0])
                add_wrapped(out[:, 1], pos + k * int(0.035 * SAMPLE_RATE), p[:, 1])
            pos += int(period * rng.uniform(0.9, 1.1) * SAMPLE_RATE)
    return out


NATURE_LAYERS = {
    "rain": rain,
    "wind": wind,
    "stream": stream,
    "ocean": ocean,
    "birds": birds,
    "crickets": crickets,
}


# ---------------------------------------------------------------------------
# Mixing and output
# ---------------------------------------------------------------------------

def normalize_rms(x, target=1.0):
    return x * (target / (np.sqrt(np.mean(x**2)) + 1e-12))


def render_mix(duration, seed, music, nature, music_gain=1.0):
    """Render the full loop.

    music: dict of render_music kwargs, or None for no music.
    nature: dict of {layer_name: gain}.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * SAMPLE_RATE))
    mix = np.zeros((n, 2))
    if music is not None:
        mix += music_gain * normalize_rms(render_music(n, rng, **music))
    for name, gain in nature.items():
        mix += gain * normalize_rms(NATURE_LAYERS[name](n, rng))
    # Consistent loudness (~-18 dBFS RMS), with a soft limiter for stray peaks.
    mix = normalize_rms(mix, 0.125)
    return 0.89 * np.tanh(mix / 0.89)


def write_wav(path, stereo):
    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())
