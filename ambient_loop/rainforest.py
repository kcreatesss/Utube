"""Animated abstract-vector Amazon rainforest interior.

The scene is drawn as flat vector shapes on every frame: layered trunks,
light shafts through the canopy, drifting mist, swaying leaves and vines,
falling leaves, rain, floating spores and a passing blue morpho butterfly.

Every motion is periodic over the loop length T: oscillations use whole
numbers of cycles per loop, and anything that travels (rain, leaves, mist,
the butterfly) crosses the frame a whole number of times, so frame T
is frame 0 again.
"""

import math
import multiprocessing as mp
import subprocess

import cairocffi as cairo
import numpy as np

TAU = 2 * math.pi


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


PALETTE = {
    "sky_top": hex_rgb("#0d3b36"),
    "sky_mid": hex_rgb("#6fae8c"),
    "glow": hex_rgb("#f6e7a8"),
    "floor": hex_rgb("#06201b"),
    "far_trunk": hex_rgb("#4f8a73"),
    "mid_trunk": hex_rgb("#1d4a3d"),
    "mid_trunk_lit": hex_rgb("#3d7a5c"),
    "liana": hex_rgb("#163b31"),
    "canopy": [hex_rgb(c) for c in ("#0f3a2f", "#14493a", "#1b5a45")],
    "fore": [hex_rgb(c) for c in ("#072620", "#0b3127", "#0f3d2e", "#134a35")],
    "vein": hex_rgb("#2e7a52"),
    "fern": [hex_rgb(c) for c in ("#0c3528", "#124431", "#1a5a3c")],
    "heliconia": [hex_rgb(c) for c in ("#e2452c", "#f07a2a", "#f5c542")],
    "morpho": [hex_rgb(c) for c in ("#1e7bd6", "#5fd3ff", "#0a1f3a")],
    "light": hex_rgb("#fff1b8"),
    "rain": hex_rgb("#d9f2ea"),
}


# ---------------------------------------------------------------------------
# Scene description (built once from a seed; rendering is a pure function of t)
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self, width, height, duration, seed=1):
        self.W, self.H, self.T = width, height, duration
        self.s = height / 1080  # all sizes are authored for 1080p
        rng = np.random.default_rng(seed)
        W, H, s = width, height, self.s

        # Integer harmonics for the global wind gusts (whole cycles per loop).
        self.gust_k = [int(round(duration / p)) or 1 for p in (41, 23, 13)]
        self.gust_phase = rng.uniform(0, TAU, 3)

        self.light_x = W * 0.58  # where sunlight breaks through the canopy

        self.far_trunks = [
            dict(x=x, w=rng.uniform(10, 26) * s, lean=rng.uniform(-0.04, 0.04),
                 alpha=rng.uniform(0.35, 0.6))
            for x in np.sort(rng.uniform(-0.05, 1.05, 16)) * W
        ]
        self.mid_trunks = [
            dict(x=x * W, w=rng.uniform(34, 60) * s, lean=rng.uniform(-0.05, 0.05),
                 curve=rng.uniform(-40, 40) * s)
            for x in (0.07, 0.27, 0.43, 0.79, 0.95)
        ]
        self.shafts = [
            dict(x=self.light_x + dx * W, top_w=rng.uniform(30, 80) * s,
                 spread=rng.uniform(1.8, 3.2), slant=rng.uniform(0.18, 0.3),
                 alpha=rng.uniform(0.12, 0.24), k=int(rng.integers(6, 14)),
                 ph=rng.uniform(0, TAU))
            for dx in (-0.16, -0.08, -0.02, 0.05, 0.12, 0.2)
        ]
        self.mist = [
            dict(x0=rng.uniform(0, 1),
                 y=rng.uniform(0.9, 1.0) * H if i >= 4 else rng.uniform(0.55, 0.9) * H,
                 rx=rng.uniform(300, 600) * s, ry=rng.uniform(70, 130 if i >= 4 else 90) * s,
                 alpha=rng.uniform(0.08, 0.18), k=int(rng.integers(1, 3)),
                 dir=rng.choice([-1, 1]), front=i >= 4)
            for i in range(7)
        ]
        self.lianas = [
            dict(x1=rng.uniform(0, 1) * W, x2=rng.uniform(0, 1) * W,
                 sag=rng.uniform(0.25, 0.6) * H, w=rng.uniform(3, 6) * s,
                 k=int(rng.integers(40, 70)), ph=rng.uniform(0, TAU))
            for _ in range(5)
        ]
        self.canopy = [
            dict(x=rng.uniform(-0.05, 1.05) * W, y=rng.uniform(-60, 20) * s,
                 L=rng.uniform(160, 300) * s, Wd=rng.uniform(50, 90) * s,
                 angle=rng.uniform(0.6, 2.5), bend=rng.uniform(-0.3, 0.3),
                 color=int(rng.integers(0, 3)), k=int(rng.integers(45, 80)),
                 ph=rng.uniform(0, TAU))
            for _ in range(26)
        ]
        self.vines = [
            dict(x=x * W, L=rng.uniform(0.3, 0.6) * H, k=int(rng.integers(35, 60)),
                 ph=rng.uniform(0, TAU), leaves=int(rng.integers(8, 14)))
            for x in (0.15, 0.36, 0.66, 0.88)
        ]
        # Big foreground leaves fanning in from the bottom corners and sides.
        self.fore_leaves = []
        for side, bx in ((-1, -0.02), (1, 1.02)):
            for i in range(7):
                self.fore_leaves.append(dict(
                    x=(bx + side * -rng.uniform(0, 0.12)) * W,
                    y=H * rng.uniform(0.85, 1.1),
                    L=rng.uniform(380, 620) * s, Wd=rng.uniform(90, 150) * s,
                    angle=(-math.pi / 2 - side * rng.uniform(0.2, 0.9)),
                    bend=side * rng.uniform(0.15, 0.45),
                    color=int(rng.integers(0, 4)), k=int(rng.integers(50, 75)),
                    ph=rng.uniform(0, TAU)))
        self.ferns = [
            dict(x=x * W, y=H * rng.uniform(1.0, 1.06), L=rng.uniform(220, 380) * s,
                 angle=-math.pi / 2 + rng.uniform(-0.9, 0.9),
                 curl=rng.uniform(0.6, 1.2) * rng.choice([-1, 1]),
                 color=int(rng.integers(0, 3)), k=int(rng.integers(55, 85)),
                 ph=rng.uniform(0, TAU))
            for x in np.linspace(0.12, 0.88, 14) + rng.uniform(-0.04, 0.04, 14)
        ]
        self.heliconias = [
            dict(x=0.22 * W, h=0.42 * H, k=int(rng.integers(50, 70)), ph=rng.uniform(0, TAU)),
            dict(x=0.84 * W, h=0.33 * H, k=int(rng.integers(50, 70)), ph=rng.uniform(0, TAU)),
        ]

        span = H + 200 * s
        self.rain_span = span
        self.rain = []
        for layer, (n, speed, length, width, alpha) in enumerate((
                (160, 700, 18, 1.0, 0.18),   # far
                (110, 1100, 34, 1.6, 0.22),  # mid
                (40, 1600, 70, 2.4, 0.18))):  # near, blurry streaks
            for _ in range(n):
                v = speed * s * rng.uniform(0.85, 1.15)
                self.rain.append(dict(
                    x=rng.uniform(-0.1, 1.1) * W, y0=rng.uniform(0, span),
                    k=max(1, int(round(v * duration / span))),
                    len=length * s, w=width * s, alpha=alpha, layer=layer))

        self.falling = [
            dict(x0=rng.uniform(0.05, 0.95) * W, y0=rng.uniform(0, 1),
                 k=int(rng.integers(1, 4)), sway=rng.uniform(40, 110) * s,
                 sk=int(round(duration / rng.uniform(7, 12))), ph=rng.uniform(0, TAU),
                 size=rng.uniform(14, 26) * s, color=int(rng.integers(0, 3)),
                 spin=int(round(duration / rng.uniform(4, 9))))
            for _ in range(9)
        ]
        self.spores = [
            dict(x=self.light_x + rng.uniform(-0.25, 0.3) * W, y=rng.uniform(0.15, 0.85) * H,
                 ax=rng.uniform(20, 70) * s, ay=rng.uniform(20, 60) * s,
                 kx=int(rng.integers(5, 15)), ky=int(rng.integers(5, 15)),
                 ph=rng.uniform(0, TAU, 3), r=rng.uniform(1.5, 3.5) * s,
                 kp=int(rng.integers(20, 40)))
            for _ in range(35)
        ]
        # The butterfly crosses the frame a whole number of times per loop.
        self.morpho = dict(k=max(1, int(round(duration / 75))), y=0.45 * H,
                           flap=int(round(duration * 2.2)))

    # -- time helpers -------------------------------------------------------

    def phase(self, t):
        return TAU * t / self.T

    def gust(self, t, x=0.0):
        """Shared wind in about [-1, 1]; x delays it so gusts travel across."""
        p = self.phase(t) - 0.8 * x / self.W
        k, ph = self.gust_k, self.gust_phase
        return (0.55 * math.sin(k[0] * p + ph[0]) + 0.3 * math.sin(k[1] * p + ph[1])
                + 0.15 * math.sin(k[2] * p + ph[2]))

    def sway(self, t, x, k, ph, amount):
        """Angle for an element: wind gust plus its own gentle oscillation."""
        return amount * (0.65 * self.gust(t, x) + 0.35 * math.sin(k * self.phase(t) + ph))


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def leaf_path(ctx, L, Wd, bend):
    """Broad leaf along +x from the origin; bend curves the tip sideways."""
    def y(x):
        return bend * L * (x / L) ** 2
    ctx.move_to(0, 0)
    ctx.curve_to(0.25 * L, y(0.25 * L) - Wd, 0.7 * L, y(0.7 * L) - 0.8 * Wd, L, y(L))
    ctx.curve_to(0.7 * L, y(0.7 * L) + 0.8 * Wd, 0.25 * L, y(0.25 * L) + Wd, 0, 0)
    ctx.close_path()


def draw_leaf(ctx, x, y, angle, L, Wd, bend, color, vein=None, alpha=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    leaf_path(ctx, L, Wd, bend)
    ctx.set_source_rgba(*color, alpha)
    ctx.fill()
    if vein is not None:
        def yy(px):
            return bend * L * (px / L) ** 2
        ctx.set_source_rgba(*vein, 0.55 * alpha)
        ctx.set_line_width(max(1.0, Wd * 0.04))
        ctx.move_to(0, 0)
        ctx.curve_to(0.35 * L, yy(0.35 * L), 0.7 * L, yy(0.7 * L), 0.95 * L, yy(0.95 * L))
        ctx.stroke()
        ctx.set_line_width(max(0.8, Wd * 0.02))
        for f in (0.25, 0.42, 0.59, 0.76):
            px = f * L
            reach = Wd * 0.75 * (1 - f * 0.6)
            for side in (-1, 1):
                ctx.move_to(px, yy(px))
                ctx.line_to(px + 0.12 * L, yy(px + 0.12 * L) + side * reach)
            ctx.stroke()
    ctx.restore()


def frond_points(x, y, angle, L, curl, sway, n=18):
    """Points along a curling frond spine; sway bends it more toward the tip."""
    pts = []
    a = angle
    seg = L / n
    for i in range(n + 1):
        pts.append((x, y, a))
        f = i / n
        a += (curl + sway * 1.6) * 0.06 * (0.5 + f)
        x += seg * math.cos(a)
        y += seg * math.sin(a)
    return pts


def draw_fern(ctx, f, scene, t):
    sw = scene.sway(t, f["x"], f["k"], f["ph"], 0.35)
    pts = frond_points(f["x"], f["y"], f["angle"], f["L"], f["curl"], sw)
    color = PALETTE["fern"][f["color"]]
    ctx.set_source_rgb(*color)
    ctx.set_line_width(3 * scene.s)
    ctx.move_to(*pts[0][:2])
    for px, py, _ in pts[1:]:
        ctx.line_to(px, py)
    ctx.stroke()
    n = len(pts)
    for i, (px, py, a) in enumerate(pts[1:-1], start=1):
        f_len = f["L"] * 0.22 * math.sin(math.pi * i / n) + 4 * scene.s
        for side in (-1, 1):
            ctx.save()
            ctx.translate(px, py)
            ctx.rotate(a + side * 1.0)
            leaf_path(ctx, f_len, f_len * 0.18, side * 0.15)
            ctx.restore()
    ctx.fill()


def draw_trunk(ctx, x, w, lean, curve, H, color, lit=None, buttress=False, s=1.0):
    top_x = x + lean * H
    ctx.move_to(x - w / 2, H)
    ctx.curve_to(x - w / 2 + curve, H * 0.6, top_x - w * 0.35 - curve, H * 0.3, top_x - w * 0.35, -10)
    ctx.line_to(top_x + w * 0.35, -10)
    ctx.curve_to(top_x + w * 0.35 - curve, H * 0.3, x + w / 2 + curve, H * 0.6, x + w / 2, H)
    ctx.close_path()
    ctx.set_source_rgba(*color)
    ctx.fill()
    if lit is not None:  # flat highlight strip on the side facing the light
        ctx.move_to(x + w * 0.15, H)
        ctx.curve_to(x + w * 0.15 + curve, H * 0.6, top_x + w * 0.05 - curve, H * 0.3, top_x + w * 0.05, -10)
        ctx.line_to(top_x + w * 0.3, -10)
        ctx.curve_to(top_x + w * 0.3 - curve, H * 0.3, x + w * 0.42 + curve, H * 0.6, x + w * 0.42, H)
        ctx.close_path()
        ctx.set_source_rgba(*lit, 0.55)
        ctx.fill()
    if buttress:  # flared roots
        ctx.set_source_rgba(*color)
        for side in (-1, 1):
            ctx.move_to(x + side * w * 0.3, H * 0.72)
            ctx.curve_to(x + side * w * 0.6, H * 0.84, x + side * w * 1.4, H * 0.86, x + side * w * 1.9, H * 0.88)
            ctx.line_to(x, H * 0.88)
            ctx.close_path()
            ctx.fill()


def draw_heliconia(ctx, h, scene, t):
    """Hanging zigzag bracts on a stem: a splash of red/orange colour."""
    s = scene.s
    sw = scene.sway(t, h["x"], h["k"], h["ph"], 0.08)
    base_x, base_y = h["x"], scene.H + 10
    top_x, top_y = base_x + sw * h["h"], base_y - h["h"]
    ctx.set_source_rgb(*PALETTE["fore"][2])
    ctx.set_line_width(6 * s)
    ctx.move_to(base_x, base_y)
    ctx.curve_to(base_x, base_y - h["h"] * 0.5, top_x, top_y + h["h"] * 0.3, top_x, top_y)
    ctx.stroke()
    red, orange, yellow = PALETTE["heliconia"]
    x, y = top_x, top_y
    for i in range(7):
        side = 1 if i % 2 else -1
        size = (34 - i * 3) * s
        ctx.move_to(x, y)
        ctx.line_to(x + side * size * 1.3, y + size * 0.2)
        ctx.line_to(x + side * size * 0.2, y + size * 0.9)
        ctx.close_path()
        ctx.set_source_rgb(*(red if i % 3 else orange))
        ctx.fill()
        ctx.move_to(x + side * size * 1.3, y + size * 0.2)
        ctx.line_to(x + side * size * 0.9, y + size * 0.05)
        ctx.line_to(x + side * size * 1.0, y + size * 0.35)
        ctx.close_path()
        ctx.set_source_rgb(*yellow)
        ctx.fill()
        y += size * 0.75
        x += sw * 12 * s
    # Two broad leaves at the base of the stem.
    for side in (-1, 1):
        draw_leaf(ctx, base_x, base_y - h["h"] * 0.15, -math.pi / 2 + side * 0.5 + sw * 1.5,
                  h["h"] * 0.75, 45 * s, side * 0.2, PALETTE["fore"][1], PALETTE["vein"])


def draw_morpho(ctx, scene, t):
    m = scene.morpho
    s = scene.s
    span = scene.W * 1.6
    u = (m["k"] * t / scene.T) % 1.0
    x = -0.3 * scene.W + u * span
    p = scene.phase(t)
    y = m["y"] + 90 * s * math.sin(m["k"] * 5 * p) + 40 * s * math.sin(m["k"] * 13 * p + 1)
    flap = abs(math.cos(m["flap"] * p))
    blue, sheen, edge = PALETTE["morpho"]
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(1.7, 1.7)
    ctx.rotate(0.15 * math.sin(m["k"] * 9 * p))
    for side in (-1, 1):
        for (wx, wy, rx, ry) in ((10, -6, 18, 12), (7, 8, 12, 9)):
            ctx.save()
            ctx.translate(side * wx * s * flap, wy * s)
            ctx.scale(max(0.08, flap) * rx * s, ry * s)
            ctx.arc(0, 0, 1, 0, TAU)
            ctx.restore()
            grad = cairo.RadialGradient(0, 0, 0, 0, 0, 22 * s)
            grad.add_color_stop_rgb(0, *sheen)
            grad.add_color_stop_rgb(0.6, *blue)
            grad.add_color_stop_rgb(1, *edge)
            ctx.set_source(grad)
            ctx.fill()
    ctx.set_source_rgb(*edge)
    ctx.set_line_width(3 * s)
    ctx.move_to(0, -10 * s)
    ctx.line_to(0, 12 * s)
    ctx.stroke()
    ctx.restore()


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(scene, t, surface=None):
    W, H, s = scene.W, scene.H, scene.s
    if surface is None:
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surface)
    p = scene.phase(t)
    pal = PALETTE

    # Background: deep canopy teal fading into a warm, misty glow.
    bg = cairo.LinearGradient(0, 0, 0, H)
    bg.add_color_stop_rgb(0, *pal["sky_top"])
    bg.add_color_stop_rgb(0.45, *pal["sky_mid"])
    bg.add_color_stop_rgb(1, *pal["floor"])
    ctx.set_source(bg)
    ctx.paint()
    glow = cairo.RadialGradient(scene.light_x, H * 0.12, 0, scene.light_x, H * 0.12, H * 0.8)
    glow.add_color_stop_rgba(0, *pal["glow"], 0.85)
    glow.add_color_stop_rgba(0.5, *pal["glow"], 0.18)
    glow.add_color_stop_rgba(1, *pal["glow"], 0)
    ctx.set_source(glow)
    ctx.paint()

    # Far trunks, hazy and nearly still.
    for tr in scene.far_trunks:
        lean = tr["lean"] + 0.004 * scene.gust(t, tr["x"])
        draw_trunk(ctx, tr["x"], tr["w"], lean, 0, H + 20 * s, (*pal["far_trunk"], tr["alpha"]))

    # Light shafts, slowly breathing.
    for sh in scene.shafts:
        a = sh["alpha"] * (0.65 + 0.35 * math.sin(sh["k"] * p + sh["ph"]))
        top_l, top_r = sh["x"] - sh["top_w"] / 2, sh["x"] + sh["top_w"] / 2
        bot_c = sh["x"] - sh["slant"] * H
        half = sh["top_w"] * sh["spread"] / 2
        ctx.move_to(top_l, -5)
        ctx.line_to(top_r, -5)
        ctx.line_to(bot_c + half, H)
        ctx.line_to(bot_c - half, H)
        ctx.close_path()
        g = cairo.LinearGradient(0, 0, 0, H)
        g.add_color_stop_rgba(0, *pal["light"], a)
        g.add_color_stop_rgba(0.7, *pal["light"], a * 0.35)
        g.add_color_stop_rgba(1, *pal["light"], 0)
        ctx.set_source(g)
        ctx.fill()

    def draw_mist(front):
        for m in scene.mist:
            if m["front"] != front:
                continue
            travel = W + 2 * m["rx"]
            x = (m["x0"] * travel + m["dir"] * m["k"] * travel * t / scene.T) % travel - m["rx"]
            ctx.save()
            ctx.translate(x, m["y"])
            ctx.scale(m["rx"], m["ry"])
            g = cairo.RadialGradient(0, 0, 0, 0, 0, 1)
            g.add_color_stop_rgba(0, 0.9, 0.97, 0.93, m["alpha"])
            g.add_color_stop_rgba(1, 0.9, 0.97, 0.93, 0)
            ctx.set_source(g)
            ctx.arc(0, 0, 1, 0, TAU)
            ctx.fill()
            ctx.restore()

    draw_mist(front=False)

    # Mid trunks with buttress roots, and lianas strung between them.
    for tr in scene.mid_trunks:
        lean = tr["lean"] + 0.006 * scene.gust(t, tr["x"])
        draw_trunk(ctx, tr["x"], tr["w"], lean, tr["curve"], H + 20 * s, (*pal["mid_trunk"], 1),
                   lit=pal["mid_trunk_lit"] if tr["x"] < scene.light_x else None, buttress=True, s=s)
    for li in scene.lianas:
        sag = li["sag"] * (1 + 0.04 * math.sin(li["k"] * p + li["ph"]))
        dx = 25 * s * scene.gust(t, li["x1"])
        ctx.move_to(li["x1"], -5)
        ctx.curve_to(li["x1"] + dx, sag, li["x2"] + dx, sag, li["x2"], -5)
        ctx.set_source_rgb(*pal["liana"])
        ctx.set_line_width(li["w"])
        ctx.stroke()

    # Forest floor: low rolling mounds that anchor the trunks.
    ctx.move_to(-10, H)
    for i in range(9):
        x0, x1 = W * i / 8, W * (i + 1) / 8
        y_mid = H * (0.86 + 0.025 * math.sin(i * 2.3))
        ctx.curve_to(x0 + (x1 - x0) * 0.3, y_mid, x0 + (x1 - x0) * 0.7, y_mid, x1, H * 0.88)
    ctx.line_to(W + 10, H)
    ctx.close_path()
    floor = cairo.LinearGradient(0, H * 0.84, 0, H)
    floor.add_color_stop_rgb(0, *pal["mid_trunk"])
    floor.add_color_stop_rgb(1, *pal["floor"])
    ctx.set_source(floor)
    ctx.fill()

    # Spores drifting in the light.
    for sp in scene.spores:
        x = sp["x"] + sp["ax"] * math.sin(sp["kx"] * p + sp["ph"][0])
        y = sp["y"] + sp["ay"] * math.sin(sp["ky"] * p + sp["ph"][1])
        a = 0.35 + 0.35 * math.sin(sp["kp"] * p + sp["ph"][2])
        g = cairo.RadialGradient(x, y, 0, x, y, sp["r"] * 3)
        g.add_color_stop_rgba(0, *pal["light"], a)
        g.add_color_stop_rgba(1, *pal["light"], 0)
        ctx.set_source(g)
        ctx.arc(x, y, sp["r"] * 3, 0, TAU)
        ctx.fill()

    # Canopy leaves hanging from the top edge.
    for c in scene.canopy:
        a = c["angle"] + scene.sway(t, c["x"], c["k"], c["ph"], 0.07)
        draw_leaf(ctx, c["x"], c["y"], a, c["L"], c["Wd"], c["bend"], pal["canopy"][c["color"]],
                  pal["vein"], alpha=0.95)

    def draw_rain(layers):
        for d in scene.rain:
            if d["layer"] not in layers:
                continue
            y = (d["y0"] + d["k"] * scene.rain_span * t / scene.T) % scene.rain_span - 100 * s
            x = d["x"] + y * 0.08
            ctx.move_to(x, y)
            ctx.line_to(x - d["len"] * 0.08, y - d["len"])
            ctx.set_source_rgba(*pal["rain"], d["alpha"])
            ctx.set_line_width(d["w"])
            ctx.stroke()

    draw_rain((0, 1))
    draw_morpho(ctx, scene, t)

    # Hanging vines: pendulums that swing more toward their tips.
    for v in scene.vines:
        sw = scene.sway(t, v["x"], v["k"], v["ph"], 0.12)
        n = 20
        x, y = v["x"], -5.0
        pts = [(x, y)]
        for i in range(n):
            a = math.pi / 2 - sw * (i / n) * 1.5
            x += v["L"] / n * math.cos(a)
            y += v["L"] / n * math.sin(a)
            pts.append((x, y))
        ctx.set_source_rgb(*pal["fore"][1])
        ctx.set_line_width(3 * s)
        ctx.move_to(*pts[0])
        for pt in pts[1:]:
            ctx.line_to(*pt)
        ctx.stroke()
        for j in range(v["leaves"]):
            i = 2 + int((n - 2) * j / v["leaves"])
            side = 1 if j % 2 else -1
            draw_leaf(ctx, *pts[i], math.pi / 2 + side * 0.9 - sw, 34 * s, 12 * s, side * 0.2,
                      pal["fore"][2 + j % 2])

    # Falling leaves, tumbling down.
    for fl in scene.falling:
        span = H + 120 * s
        y = (fl["y0"] * span + fl["k"] * span * t / scene.T) % span - 60 * s
        x = fl["x0"] + fl["sway"] * math.sin(fl["sk"] * p + fl["ph"])
        rot = 0.9 * math.sin(fl["spin"] * p + fl["ph"]) + math.pi / 2
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(1, 0.55 + 0.45 * abs(math.cos(fl["spin"] * p)))
        draw_leaf(ctx, 0, 0, rot, fl["size"], fl["size"] * 0.4, 0.15,
                  pal["heliconia"][fl["color"]] if fl["color"] == 2 else pal["canopy"][2], alpha=0.9)
        ctx.restore()

    # Foreground: ferns, heliconias and the big framing leaves.
    for f in scene.ferns:
        draw_fern(ctx, f, scene, t)
    for h in scene.heliconias:
        draw_heliconia(ctx, h, scene, t)
    for lf in scene.fore_leaves:
        a = lf["angle"] + scene.sway(t, lf["x"], lf["k"], lf["ph"], 0.05)
        draw_leaf(ctx, lf["x"], lf["y"], a, lf["L"], lf["Wd"], lf["bend"],
                  pal["fore"][lf["color"]], pal["vein"])

    draw_rain((2,))
    draw_mist(front=True)

    # Vignette.
    vg = cairo.RadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.75)
    vg.add_color_stop_rgba(0, 0, 0, 0, 0)
    vg.add_color_stop_rgba(1, 0, 0, 0, 0.55)
    ctx.set_source(vg)
    ctx.paint()

    surface.flush()
    return surface


# ---------------------------------------------------------------------------
# Video encoding (frames rendered in parallel, piped to ffmpeg)
# ---------------------------------------------------------------------------

_worker = {}


def _init_worker(width, height, duration, seed):
    _worker["scene"] = Scene(width, height, duration, seed)
    _worker["surface"] = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)


def _render_bytes(t):
    surface = render_frame(_worker["scene"], t, _worker["surface"])
    return bytes(surface.get_data())


def save_still(path, width, height, duration, t=0.0, seed=1):
    render_frame(Scene(width, height, duration, seed), t).write_to_png(str(path))


def build_video(audio_path, out_path, duration, width, height, fps, seed=1, workers=None):
    n_frames = int(round(duration * fps))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
        # Cairo RGB24 is native-endian 0xXXRRGGBB, i.e. BGRx bytes on x86/ARM.
        "-f", "rawvideo", "-pix_fmt", "bgr0", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a", "-t", f"{duration}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    times = (i / fps for i in range(n_frames))
    with mp.Pool(workers, initializer=_init_worker, initargs=(width, height, duration, seed)) as pool:
        for frame in pool.imap(_render_bytes, times, chunksize=8):
            proc.stdin.write(frame)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
