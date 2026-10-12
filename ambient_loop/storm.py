"""Shared lightning timeline, so the flashes on screen and the thunder line up."""

import math

import numpy as np

THUNDER_DELAY = 2.6  # seconds between a flash and its thunder (a distant storm)
FLASH_SECONDS = 1.5


def lightning_times(duration, seed, strikes_per_minute=0.8):
    """Strike times spread through the loop, one per equal slice, jittered."""
    rng = np.random.default_rng(seed + 7919)
    k = max(1, round(duration / 60 * strikes_per_minute))
    return [(i + rng.uniform(0.15, 0.85)) * duration / k for i in range(k)]


def flash_intensity(dt):
    """Brightness 0..1 at dt seconds after a strike: a triple flicker then afterglow."""
    if dt < 0 or dt > FLASH_SECONDS:
        return 0.0
    f = 0.25 * math.exp(-dt / 0.35)
    for start, amp in ((0.0, 1.0), (0.09, 0.55), (0.24, 0.85)):
        if dt >= start:
            f += amp * math.exp(-(dt - start) / 0.06)
    return min(1.0, f)
