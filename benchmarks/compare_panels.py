"""Pixel diff of two rendered Figure-13 panels (figures/drop_1m.py output).

    python benchmarks/compare_panels.py figures/img_drop_1m_4060_fp64.png figures/img_drop_1m_4060.png

Reports the fraction of bit-identical pixels and the max per-channel delta -- the
comparison artifacts/fig13_mac_report.md sec 4 made (99.86 % identical, max 3/255).
"""
import sys
import numpy as np
from PIL import Image

a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.int16)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB")).astype(np.int16)
assert a.shape == b.shape, (a.shape, b.shape)
d = np.abs(a - b)
same = float((d.max(axis=2) == 0).mean())
print(f"shape {a.shape[:2]}  identical pixels {100*same:.3f}%  "
      f"max channel delta {int(d.max())}/255  pixels >4/255: {int((d.max(axis=2) > 4).sum())}")
