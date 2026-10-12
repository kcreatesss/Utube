"""Overlay a "liquid glass" countdown timer on a finished video.

The glass panel is built in ffmpeg so it reacts to the live video behind it:
the backdrop is blurred, brightened and saturated, then refracted through a
lens-edge displacement map (content bends near the rim, like Apple's Liquid
Glass). A per-second Cairo layer adds the rim highlights, soft shadow,
tabular digits and a progress bar.

    python -m ambient_loop.timer input.mp4 output.mp4 --minutes 10
"""

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cairocffi as cairo
import numpy as np

FONT = "Inter Display"
PAD = 60  # room around the panel for refraction sampling and the shadow


def rounded_rect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def rounded_rect_sdf(pw, ph, bw, bh, r):
    """Signed distance (px) to the panel edge over the padded area; negative inside."""
    y, x = np.mgrid[0:ph, 0:pw].astype(float)
    qx = np.abs(x + 0.5 - pw / 2) - (bw / 2 - r)
    qy = np.abs(y + 0.5 - ph / 2) - (bh / 2 - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - r


def write_gray_png(path, values):
    """Save a 0-255 array as a (grey RGB) PNG through Cairo."""
    v = np.clip(np.round(values), 0, 255).astype(np.uint32)
    h, w = v.shape
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    buf = np.frombuffer(surface.get_data(), dtype=np.uint32).reshape(h, surface.get_stride() // 4)
    buf[:, :w] = (v << 16) | (v << 8) | v
    surface.mark_dirty()
    surface.write_to_png(str(path))


def make_glass_maps(folder, bw, bh, r, rim=56, bend=44, zoom=0.04):
    """Displacement maps (x, y) and the alpha mask for the padded panel area."""
    pw, ph = bw + 2 * PAD, bh + 2 * PAD
    d = rounded_rect_sdf(pw, ph, bw, bh, r)
    gy, gx = np.gradient(d)  # points outward
    norm = np.hypot(gx, gy) + 1e-9
    nx, ny = gx / norm, gy / norm
    depth = np.clip(-d, 0, None)
    edge = np.clip(1 - depth / rim, 0, 1) ** 2.2 * (d < 0)
    y, x = np.mgrid[0:ph, 0:pw].astype(float)
    # Near the rim, sample from further out (the lens bends the backdrop);
    # across the face, sample slightly toward the centre (gentle magnification).
    dx = nx * edge * bend - (x - pw / 2) * zoom * (d < 0)
    dy = ny * edge * bend - (y - ph / 2) * zoom * (d < 0)
    write_gray_png(folder / "xmap.png", 128 + dx)
    write_gray_png(folder / "ymap.png", 128 + dy)
    write_gray_png(folder / "mask.png", np.clip(0.5 - d, 0, 1) * 255)


def text_width(ctx, s):
    return ctx.text_extents(s)[4]  # x_advance


def draw_overlay(path, bw, bh, r, remaining, total):
    pw, ph = bw + 2 * PAD, bh + 2 * PAD
    x0, y0 = PAD, PAD
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, pw, ph)
    ctx = cairo.Context(surface)

    # Soft shadow, outside the glass only.
    ctx.save()
    ctx.rectangle(0, 0, pw, ph)
    rounded_rect(ctx, x0, y0, bw, bh, r)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.clip()
    for i in range(14):
        grow = i * 3
        rounded_rect(ctx, x0 - grow, y0 - grow + 12, bw + 2 * grow, bh + 2 * grow, r + grow)
        ctx.set_source_rgba(0, 0.03, 0.02, 0.022)
        ctx.fill()
    ctx.restore()

    # Glass body: faint white tint, brighter toward the top, and a top glare.
    ctx.save()
    rounded_rect(ctx, x0, y0, bw, bh, r)
    ctx.clip()
    tint = cairo.LinearGradient(0, y0, 0, y0 + bh)
    tint.add_color_stop_rgba(0, 1, 1, 1, 0.12)
    tint.add_color_stop_rgba(0.55, 1, 1, 1, 0.02)
    tint.add_color_stop_rgba(1, 1, 1, 1, 0.06)
    ctx.set_source(tint)
    ctx.paint()
    glare = cairo.RadialGradient(x0 + bw * 0.28, y0 - bh * 0.1, 0, x0 + bw * 0.28, y0 - bh * 0.1, bw * 0.55)
    glare.add_color_stop_rgba(0, 1, 1, 1, 0.22)
    glare.add_color_stop_rgba(1, 1, 1, 1, 0)
    ctx.set_source(glare)
    ctx.paint()
    ctx.restore()

    # Specular rim: bright top-left, dimmer sides, a second glint bottom-right.
    rim = cairo.LinearGradient(x0, y0, x0 + bw, y0 + bh)
    rim.add_color_stop_rgba(0, 1, 1, 1, 0.95)
    rim.add_color_stop_rgba(0.35, 1, 1, 1, 0.25)
    rim.add_color_stop_rgba(0.65, 1, 1, 1, 0.12)
    rim.add_color_stop_rgba(1, 1, 1, 1, 0.7)
    rounded_rect(ctx, x0 + 1, y0 + 1, bw - 2, bh - 2, r - 1)
    ctx.set_source(rim)
    ctx.set_line_width(2.2)
    ctx.stroke()
    rounded_rect(ctx, x0 + 4, y0 + 4, bw - 8, bh - 8, r - 4)
    ctx.set_source_rgba(1, 1, 1, 0.12)
    ctx.set_line_width(1.2)
    ctx.stroke()

    cx = x0 + bw / 2

    # Label.
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(26)
    label = "F O C U S"
    ctx.move_to(cx - text_width(ctx, label) / 2, y0 + 66)
    ctx.set_source_rgba(1, 1, 1, 0.78)
    ctx.show_text(label)

    # Digits in fixed-width cells so they don't shuffle as they change.
    mins, secs = divmod(remaining, 60)
    text = f"{mins:02d}:{secs:02d}"
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(150)
    cell = max(text_width(ctx, str(n)) for n in range(10))
    colon = text_width(ctx, ":") * 1.15
    total_w = 4 * cell + colon
    baseline = y0 + 205
    for dx_shadow, alpha, colour in ((3, 0.22, (0, 0.05, 0.03)), (0, 0.96, (1, 1, 1))):
        x = cx - total_w / 2
        for ch in text:
            w = colon if ch == ":" else cell
            ctx.move_to(x + (w - text_width(ctx, ch)) / 2, baseline + dx_shadow)
            ctx.set_source_rgba(*colour, alpha)
            ctx.show_text(ch)
            x += w

    # Progress capsule.
    track_w, track_h = bw * 0.62, 8
    tx, ty = cx - track_w / 2, y0 + bh - 52
    rounded_rect(ctx, tx, ty, track_w, track_h, track_h / 2)
    ctx.set_source_rgba(1, 1, 1, 0.22)
    ctx.fill()
    fill_w = max(track_h, track_w * remaining / total)
    rounded_rect(ctx, tx, ty, fill_w, track_h, track_h / 2)
    ctx.set_source_rgba(1, 1, 1, 0.9)
    ctx.fill()

    surface.write_to_png(str(path))


def add_timer(src, dst, minutes=10, box=(760, 300), radius=96, width=1920, height=1080,
              hevc_kbps=None, still_at=None):
    total = int(minutes * 60)
    bw, bh = box
    assert bw * bh <= 0.25 * width * height, "timer must stay within 25% of the frame"
    px, py = (width - bw) // 2 - PAD, (height - bh) // 2 - PAD
    pw, ph = bw + 2 * PAD, bh + 2 * PAD
    work = Path(tempfile.mkdtemp(prefix="glass_timer_"))
    try:
        make_glass_maps(work, bw, bh, radius)
        seconds = [int(still_at)] if still_at is not None else range(total + 1)
        for k in seconds:
            draw_overlay(work / f"ov_{k:04d}.png", bw, bh, radius, max(0, total - k), total)

        graph = (
            f"[0:v]split=2[base][c];"
            f"[c]crop={pw}:{ph}:{px}:{py},gblur=sigma=9,eq=brightness=0.05:saturation=1.4[blur];"
            f"[blur][1:v][2:v]displace=edge=smear[ref];"
            f"[3:v]format=gray[mask];[ref][mask]alphamerge[glass];"
            f"[base][glass]overlay={px}:{py}[g1];"
            f"[g1][4:v]overlay={px}:{py}:format=auto,format=yuv420p[out]"
        )
        loops = []
        for name in ("xmap", "ymap", "mask"):
            loops += ["-loop", "1", "-framerate", "30", "-i", str(work / f"{name}.png")]
        if still_at is not None:
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(still_at), "-i", str(src), *loops,
                   "-loop", "1", "-i", str(work / f"ov_{int(still_at):04d}.png"),
                   "-filter_complex", graph, "-map", "[out]", "-frames:v", "1", str(dst)]
            subprocess.run(cmd, check=True)
            return
        inputs = ["-i", str(src), *loops, "-framerate", "1", "-i", str(work / "ov_%04d.png")]
        common = ["-filter_complex", graph, "-map", "[out]", "-t", str(total)]
        if hevc_kbps:
            x265 = ["-c:v", "libx265", "-preset", "fast", "-b:v", f"{hevc_kbps}k", "-tag:v", "hvc1"]
            stats = str(work / "x265.log")
            subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", *inputs, *common, *x265,
                            "-x265-params", f"pass=1:stats={stats}:log-level=error", "-an", "-f", "mp4",
                            "/dev/null"], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", *inputs, *common, *x265,
                            "-x265-params", f"pass=2:stats={stats}:log-level=error", "-map", "0:a",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)], check=True)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", *inputs, *common,
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-map", "0:a",
                            "-c:a", "copy", "-movflags", "+faststart", str(dst)], check=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(prog="python -m ambient_loop.timer",
                                description="Overlay a liquid-glass countdown timer on a video.")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--hevc-kbps", type=int, help="encode HEVC at this video bitrate (2-pass), e.g. 1080")
    p.add_argument("--still", type=float, metavar="T", help="only write a PNG of the frame at T seconds")
    args = p.parse_args()
    add_timer(args.src, args.dst, args.minutes, hevc_kbps=args.hevc_kbps, still_at=args.still)


if __name__ == "__main__":
    main()
