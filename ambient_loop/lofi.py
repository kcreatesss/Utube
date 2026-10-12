"""Jazzy lofi music: swung boom-bap drums, Rhodes chords, bass, melody, vinyl.

The tempo is nudged so a whole number of 8-bar sections fills the loop
exactly, so the beat runs straight through the loop point. Notes are placed
with add_wrapped, so tails that run past the end land at the start.
"""

import numpy as np

from .audio import (SAMPLE_RATE, SCALES, add_wrapped, circular_reverb, midi_to_hz,
                    normalize_rms, pan, root_midi, shaped_noise)

SR = SAMPLE_RATE

# ii9 - V13 - Imaj9 - vi9, as (bass degree, rootless voicing) in semitones from the key root.
PROGRESSION = [
    (2, [5, 9, 12, 16]),
    (7, [5, 9, 11, 16]),
    (0, [4, 7, 11, 14]),
    (9, [0, 4, 7, 11]),
]


# ---------------------------------------------------------------------------
# Instruments (one-shot sounds)
# ---------------------------------------------------------------------------

def _t(seconds):
    return np.arange(int(seconds * SR)) / SR


def _band_noise(rng, seconds, low, high):
    m = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(m))
    f = np.maximum(np.fft.rfftfreq(m, 1 / SR), 1)
    spec *= 1 / (1 + (low / f) ** 4) / (1 + (f / high) ** 4)
    x = np.fft.irfft(spec, m)
    return x / (np.abs(x).max() + 1e-12)


def make_kit(rng):
    t = _t(0.45)
    pitch = 46 + 85 * np.exp(-t / 0.035)
    kick = np.sin(2 * np.pi * np.cumsum(pitch) / SR) * np.exp(-t / 0.22)
    kick[:150] += _band_noise(rng, 0.45, 1000, 6000)[:150] * 0.3  # beater click

    t = _t(0.3)
    snare = (_band_noise(rng, 0.3, 900, 7000) * np.exp(-t / 0.075)
             + 0.6 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05))

    t = _t(0.08)
    rim = (np.sin(2 * np.pi * 820 * t) * 0.7 + _band_noise(rng, 0.08, 1500, 6000) * 0.5) * np.exp(-t / 0.018)

    t = _t(0.1)
    hat = _band_noise(rng, 0.1, 7000, 14000) * np.exp(-t / 0.016)
    t = _t(0.35)
    open_hat = _band_noise(rng, 0.35, 6500, 14000) * np.exp(-t / 0.11)
    return dict(kick=kick, snare=snare, rim=rim, hat=hat, open_hat=open_hat)


def rhodes(f, seconds, velocity=1.0):
    """FM electric piano: bell-like attack that mellows, with a gentle auto-pan."""
    t = _t(seconds + 1.2)
    index = (1.4 * velocity) * np.exp(-t / 0.3) + 0.25
    tone = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * f * t))
    tone += 0.12 * np.sin(2 * np.pi * 7 * f * t) * np.exp(-t / 0.04)  # tine ping
    env = np.minimum(t / 0.004, 1) * np.exp(-t / 2.0)
    release = np.clip((seconds + 1.2 - t) / 1.2, 0, 1)  # let go after `seconds`
    release = np.where(t < seconds, 1.0, release)
    mono = tone * env * release * velocity
    trem = 0.25 * np.sin(2 * np.pi * 4.2 * t)
    return np.stack([mono * (1 + trem), mono * (1 - trem)], axis=1) * 0.7


def bass(f, seconds, velocity=1.0):
    t = _t(seconds + 0.15)
    tone = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
    env = np.minimum(t / 0.012, 1) * np.exp(-t / 0.9)
    env *= np.clip((seconds + 0.15 - t) / 0.15, 0, 1)
    return tone * env * velocity


# ---------------------------------------------------------------------------
# Arrangement
# ---------------------------------------------------------------------------

def section_role(i):
    """Cycles intro -> full beat with a breakdown every few sections."""
    if i == 0:
        return "intro"  # keys + bass only; the loop flows into it from the last section
    if i % 6 == 3:
        return "break"  # rimshot + hats, no kick, airier keys
    return "full"


def render_lofi(n, rng, key="F", bpm=76, swing=0.12, mood="major"):
    beat_target = 60 / bpm
    sections = max(1, round(n / SR / (beat_target * 32)))
    bars = sections * 8
    bar_len = n / bars
    step = bar_len / 16  # 16th note, in samples

    def at(bar, s16, humanize=0.006):
        """Sample position of a 16th step in a bar, with swing and a little looseness."""
        pos = bar * bar_len + s16 * step
        if s16 % 4 == 2:  # swung 8th off-beats
            pos += swing * 4 * step
        elif s16 % 2 == 1:  # 16ths sit later still
            pos += swing * 2.5 * step
        return int(pos + rng.normal(0, humanize) * SR)

    kit = make_kit(rng)
    drums = np.zeros((n, 2))
    keys = np.zeros((n, 2))
    low = np.zeros(n)
    duck = np.zeros(n)
    root = root_midi(key)
    duck_shape = np.exp(-_t(0.4) / 0.12)

    def hit(sound, pos, gain, position=0.0):
        st = pan(sound * gain, position)
        add_wrapped(drums[:, 0], pos, st[:, 0])
        add_wrapped(drums[:, 1], pos, st[:, 1])

    scale = SCALES[mood]
    for bar in range(bars):
        section = bar // 8
        role = section_role(section)
        next_role = section_role((section + 1) % sections)
        last_bar = bar % 8 == 7
        bass_deg, voicing = PROGRESSION[bar % len(PROGRESSION)]

        # --- drums ---
        if role != "intro":
            for s16 in range(0, 16, 2):
                accent = 0.9 if s16 % 4 == 0 else 0.6
                if role == "full" and s16 == 14 and bar % 4 == 3:
                    hit(kit["open_hat"], at(bar, s16), 0.35, 0.3)
                else:
                    hit(kit["hat"], at(bar, s16), 0.32 * accent * rng.uniform(0.85, 1.1), 0.3)
            for s16 in (3, 11):  # ghost 16ths
                if rng.random() < 0.35:
                    hit(kit["hat"], at(bar, s16), 0.15, 0.3)
            if role == "full":
                kicks = [0, 10] + ([7] if bar % 2 else []) + ([15] if rng.random() < 0.2 else [])
                for s16 in kicks:
                    hit(kit["kick"], at(bar, s16), 1.0 if s16 == 0 else 0.8)
                    add_wrapped(duck, at(bar, s16, 0), duck_shape)
                for s16 in (4, 12):
                    hit(kit["snare"], at(bar, s16), 0.55 * rng.uniform(0.9, 1.05), -0.05)
                if rng.random() < 0.3:
                    hit(kit["snare"], at(bar, 15), 0.12, -0.05)  # ghost snare
            else:
                for s16 in (4, 12):
                    hit(kit["rim"], at(bar, s16), 0.35, -0.15)
            if last_bar and next_role != role and role == "full":
                for s16 in (13, 14, 15):  # little fill into the next section
                    hit(kit["snare"], at(bar, s16), 0.25 + 0.1 * (s16 - 13), -0.05)

        # --- Rhodes chords ---
        hits = [(0, 15, 0.8)] if role == "intro" else [(0, 6, 0.75), (6, 10, 0.5)] if bar % 2 else [(0, 10, 0.75), (10, 6, 0.5)]
        for s16, length, vel in hits:
            for i, semi in enumerate(voicing):
                f = midi_to_hz(root + semi)
                note = rhodes(f, length * step / SR, vel * rng.uniform(0.85, 1.0))
                pos = at(bar, s16) + int(i * 0.012 * SR)  # soft strum
                add_wrapped(keys[:, 0], pos, note[:, 0])
                add_wrapped(keys[:, 1], pos, note[:, 1])

        # --- bass ---
        b = midi_to_hz(root - 12 + bass_deg)
        if role == "intro":
            notes = [(0, 14, b)]
        else:
            fifth = midi_to_hz(root - 12 + bass_deg + 7)
            notes = [(0, 6, b), (7, 2, fifth if bar % 2 else b * 2), (10, 5, b)]
        for s16, length, f in notes:
            add_wrapped(low, at(bar, s16, 0.004), bass(f, length * step / SR, 0.9))

        # --- melody: a short Rhodes motif every other bar in the beat sections ---
        if role != "intro" and bar % 2 == 1 and rng.random() < 0.75:
            s16 = int(rng.choice([0, 2, 4]))
            idx = int(rng.integers(0, len(scale)))
            while s16 < 16:
                f = midi_to_hz(root + 12 + scale[idx % len(scale)] + 12 * (idx // len(scale)))
                note = rhodes(f, 1.5 * step / SR, 0.55 * rng.uniform(0.8, 1.0))
                p = pan(note[:, 0], 0.35)
                add_wrapped(keys[:, 0], at(bar, s16), p[:, 0])
                add_wrapped(keys[:, 1], at(bar, s16), p[:, 1])
                s16 += int(rng.choice([2, 2, 3, 4]))
                idx = max(0, idx + int(rng.choice([-1, 1, 1, 2])))

    # Sidechain: keys and bass breathe with the kick.
    pump = 1 - 0.3 * np.minimum(duck, 1.0)
    keys *= pump[:, None]
    low *= pump

    keys = circular_reverb(normalize_rms(keys), rng, seconds=2.5, wet=0.3)
    mix = (normalize_rms(drums) * 0.8 + keys * 0.9 + pan(normalize_rms(low), 0) * 0.75)

    # Lofi colour: roll off the top end, gentle tape saturation, vinyl.
    freqs = np.fft.rfftfreq(n, 1 / SR)
    tone = 1 / (1 + (freqs / 4200) ** 2) / (1 + (35 / np.maximum(freqs, 1)) ** 4)
    for ch in range(2):
        mix[:, ch] = np.fft.irfft(np.fft.rfft(mix[:, ch]) * tone, n)
    mix = np.tanh(normalize_rms(mix, 0.3) * 1.4)

    crackle = np.zeros((n, 2))
    pops = rng.integers(0, n, size=int(n / SR * 6))
    click = np.exp(-_t(0.002) / 0.0004)
    for pos in pops:
        add_wrapped(crackle[:, int(rng.integers(0, 2))], pos, click * rng.uniform(0.2, 1.0) * rng.choice([-1, 1]))
    hiss = np.stack([shaped_noise(n, rng, 800, 9000, tilt=-3) for _ in range(2)], axis=1)
    return normalize_rms(mix) + 0.06 * normalize_rms(crackle) + 0.035 * hiss
