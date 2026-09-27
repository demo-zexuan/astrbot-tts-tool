"""Generate the plugin logo (``logo.png``).

Dev-only helper: AstrBot reads ``logo.png`` from the plugin directory for the
plugin market and the WebUI card. It is not imported at runtime, so Pillow is
deliberately *not* listed in ``requirements.txt``.

Usage::

    python tools/make_logo.py            # writes logo.png next to main.py
    python tools/make_logo.py out.png    # writes somewhere else

The mark is drawn at 4x and downsampled, which keeps the rounded corners and the
waveform bars smooth at the 256x256 size AstrBot expects.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
SUPERSAMPLE = 4

# Brand palette: an indigo -> violet gradient with a white speech bubble.
GRADIENT_TOP = (79, 70, 229)
GRADIENT_BOTTOM = (139, 92, 246)
BUBBLE = (255, 255, 255, 255)


def linear_gradient(
    size: int,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> Image.Image:
    """Build a vertical linear gradient image."""
    gradient = Image.new("RGBA", (1, size))
    for y in range(size):
        ratio = y / max(size - 1, 1)
        gradient.putpixel(
            (0, y),
            (
                round(top[0] + (bottom[0] - top[0]) * ratio),
                round(top[1] + (bottom[1] - top[1]) * ratio),
                round(top[2] + (bottom[2] - top[2]) * ratio),
                255,
            ),
        )
    return gradient.resize((size, size), Image.Resampling.NEAREST)


def build_logo(size: int = SIZE) -> Image.Image:
    """Render the logo at ``size`` x ``size`` pixels."""
    scale = SUPERSAMPLE
    canvas_size = size * scale
    unit = scale  # 1 design pixel == ``scale`` device pixels

    base = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)

    # I. Rounded gradient plate
    radius = 52 * unit
    plate_mask = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(plate_mask).rounded_rectangle(
        (16 * unit, 16 * unit, 240 * unit, 240 * unit),
        radius=radius,
        fill=255,
    )
    base.paste(linear_gradient(canvas_size, GRADIENT_TOP, GRADIENT_BOTTOM), (0, 0), plate_mask)

    # II. Speech bubble
    # 1. Rounded body
    body = (58 * unit, 66 * unit, 198 * unit, 162 * unit)
    draw.rounded_rectangle(body, radius=30 * unit, fill=BUBBLE)

    # 2. Tail pointing down-left, drawn as a triangle that overlaps the body so
    #    the two shapes merge without a visible seam.
    draw.polygon(
        [
            (84 * unit, 150 * unit),
            (72 * unit, 194 * unit),
            (116 * unit, 158 * unit),
        ],
        fill=BUBBLE,
    )

    # III. Waveform bars inside the bubble
    # 1. Heights are symmetric around the vertical centre so the mark reads as a
    #    calm audio level rather than a random bar chart.
    heights = (26, 46, 58, 46, 26)
    bar_width = 13
    gap = 8
    total = len(heights) * bar_width + (len(heights) - 1) * gap
    start_x = (256 - total) / 2
    center_y = 112

    bar_colour = (99, 91, 255, 255)
    for index, height in enumerate(heights):
        left = (start_x + index * (bar_width + gap)) * unit
        top = (center_y - height / 2) * unit
        right = left + bar_width * unit
        bottom = top + height * unit
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=(bar_width / 2) * unit,
            fill=bar_colour,
        )

    # IV. Downsample for antialiasing
    return base.resize((size, size), Image.Resampling.LANCZOS)


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "logo.png"
    logo = build_logo()
    logo.save(target, format="PNG", optimize=True)
    print(f"wrote {target} ({target.stat().st_size} bytes, {logo.size[0]}x{logo.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
