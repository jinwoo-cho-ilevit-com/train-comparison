"""Deterministic probe inputs.

Generated rather than downloaded so the visual-token count measured for each model
comes from a byte-identical image, and so a probe needs no dataset access to run.
"""

from __future__ import annotations

from typing import Any

# Fixed so the per-model visual token counts are directly comparable. All three
# models use patch_size 16 but differ in spatial merge / pooling, which is exactly
# what this measures.
PROBE_IMAGE_SIZE = (448, 448)

PROBE_PAIRS = [
    ("what is shown in the image", "a square gradient test pattern"),
    ("describe the picture", "a synthetic image used for probing"),
]


def probe_image() -> Any:
    """A fixed RGB gradient. Imported lazily: pillow belongs to whatever stack the
    framework image pulled in."""
    from PIL import Image

    width, height = PROBE_IMAGE_SIZE
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return image
