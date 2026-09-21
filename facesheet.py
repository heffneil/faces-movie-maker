#!/usr/bin/env python3
"""One photo -> a 9-viseme phoneme sheet, without an image generator.

Two stages, because neither alone is enough:

1. Cartoonify. Painting mouths onto a photographic or 3D-rendered face always
   looks pasted on, because flat shapes sit badly over smooth shading. An
   edge-preserving filter flattens the shading while keeping the features
   (colour quantisation was tried first and destroyed glasses and eyes), and
   the background is keyed to white so cells isolate cleanly.

2. Synthesise the nine mouths. The lip line is found as the darkest row
   between nose and collar; the jaw is stretched downward for open poses —
   confined to the jaw, or it drags the hair and silhouette with it — and a
   shaped mouth is drawn with a shaded interior, upper teeth and a tongue.

The result is a sheet `sprite.py` can animate directly. It is synthesis, not
artwork: good enough to animate, not as good as nine drawn poses.

Usage:
    .venv/bin/python facesheet.py face.png --out sheet.png
    .venv/bin/python facesheet.py face.png --out sheet.png --no-labels --sat 1.4
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

# open, width, roundness, teeth, tongue
POSES = {
 'AI':  (1.00, 0.95, 0.05, 0.42, 0.30),
 'E':   (0.50, 1.14, 0.00, 0.80, 0.00),
 'FV':  (0.17, 0.94, 0.00, 0.95, 0.00),
 'L':   (0.56, 0.82, 0.10, 0.40, 0.75),
 'MBP': (0.00, 1.00, 0.00, 0.00, 0.00),
 'O':   (0.72, 0.62, 0.85, 0.12, 0.00),
 'U':   (0.50, 0.52, 0.95, 0.00, 0.00),
 'WQ':  (0.34, 0.48, 1.00, 0.00, 0.00),
 'REST':(0.11, 0.98, 0.00, 0.00, 0.00),
}
ORDER = ['AI', 'E', 'FV', 'L', 'MBP', 'O', 'U', 'WQ', 'REST']


def cartoonify(bgr, scale=5, sat=1.25, bg_tol=30):
    """Flatten shading, key the background to white. Returns (rgb, fg mask)."""
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    toon = cv2.edgePreservingFilter(img, flags=1, sigma_s=64, sigma_r=0.4)
    hsv = cv2.cvtColor(toon, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    toon = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    h, w = toon.shape[:2]

    ring = np.zeros((h, w), bool)
    b = max(3, int(min(h, w) * 0.02))
    ring[:b, :] = ring[-b:, :] = ring[:, :b] = ring[:, -b:] = True
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, bgc = cv2.kmeans(toon[ring].astype(np.float32), 4, None, crit, 4,
                           cv2.KMEANS_PP_CENTERS)
    dist = np.min(np.linalg.norm(toon.astype(np.float32)[:, :, None, :]
                                 - bgc[None, None], axis=3), axis=2)
    m = cv2.morphologyEx((dist < bg_tol).astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((9, 9), np.uint8))
    _, lbb, _, _ = cv2.connectedComponentsWithStats(m, 8)
    fg = (~np.isin(lbb, list(set(lbb[ring].tolist()) - {0}))).astype(np.uint8)
    nf, lf, stf, _ = cv2.connectedComponentsWithStats(fg, 8)
    if nf > 1:
        fg = (lf == 1 + int(np.argmax(stf[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    fg = ndi.binary_fill_holes(cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                                                np.ones((15, 15), np.uint8)) > 0)
    out = toon.copy()
    out[~fg] = 255
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32), fg


def find_mouth(rgb):
    """Lip line = darkest row between the nose and the collar."""
    h, w = rgb.shape[:2]
    y0, y1 = int(h * 0.66), int(h * 0.80)
    cx0 = w // 2
    prof = rgb[y0:y1, cx0 - 90:cx0 + 90].mean(axis=(1, 2))
    lip = y0 + int(np.argmin(prof))
    seg = rgb[lip - 3:lip + 4, cx0 - 110:cx0 + 110].mean(axis=(0, 2))
    wgt = np.clip(seg.max() - seg, 0, None)
    cx = int(cx0 - 110 + (np.arange(len(seg)) * wgt).sum() / max(wgt.sum(), 1e-6))
    return lip, cx, int(w * 0.085)


def build(rgb, lip, cx, half, maxd_frac=0.105):
    h, w = rgb.shape[:2]
    skin = rgb[lip - 95:lip - 65, cx + 75:cx + 115].mean(axis=(0, 1))
    LIP = tuple((skin * 0.62).astype(int))
    DARK, MID = (58, 26, 30), (104, 46, 50)
    TEETH = tuple(np.clip(skin * 0.35 + 168, 0, 255).astype(int))
    TONG = (176, 92, 96)
    MAXD = int(h * maxd_frac)

    def pose(op, wf, rnd, th, tg):
        drop = MAXD * op
        arr = rgb
        if drop > 0.5:
            k = (h - lip) / (h - lip + drop)
            ys = np.arange(h)
            sy = np.where(ys > lip, lip + (ys - lip) * k, ys)
            i0 = np.clip(sy.astype(int), 0, h - 1)
            i1 = np.clip(i0 + 1, 0, h - 1)
            t = (sy - i0)[:, None, None]
            warped = rgb[i0] * (1 - t) + rgb[i1] * t
            # keep the drop on the jaw; warping the whole frame drags the hair
            xs = np.abs(np.arange(w) - cx) / (half * 2.6)
            mx = 0.5 * (1 + np.cos(np.pi * np.clip(xs, 0, 1)))
            yv = np.clip((np.arange(h) - (lip - 22)) / 46.0, 0, 1)
            mask = (mx[None, :] * yv[:, None])[..., None]
            arr = rgb * (1 - mask) + warped * mask
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if drop < 2:
            return out

        mw, mh = half * wf, drop * 0.55
        my = lip + drop * 0.30
        box = [cx - mw, my - mh, cx + mw, my + mh]
        inner = Image.new('L', (w, h), 0)
        ImageDraw.Draw(inner).ellipse(box, fill=255)
        grad = np.zeros((h, w, 3), np.float32)
        yy = np.clip((np.arange(h) - (my - mh)) / max(2 * mh, 1), 0, 1)[:, None, None]
        grad[:] = np.array(DARK) + (np.array(MID) - np.array(DARK)) * yy
        lay = Image.fromarray(grad.astype(np.uint8))
        d = ImageDraw.Draw(lay)
        if tg > 0:
            d.ellipse([cx - mw * 0.62, my + mh - mh * 1.5 * tg,
                       cx + mw * 0.62, my + mh + mh * 0.4], fill=TONG)
        if th > 0:
            d.ellipse([cx - mw * 0.98, my - mh - mh * 0.55,
                       cx + mw * 0.98, my - mh + mh * 1.25 * th], fill=TEETH)
        out = Image.composite(lay.filter(ImageFilter.GaussianBlur(1.2)), out,
                              inner.filter(ImageFilter.GaussianBlur(1.6)))
        ring_m = Image.new('L', (w, h), 0)
        ImageDraw.Draw(ring_m).ellipse(box, outline=255,
                                       width=max(3, int(w * 0.006)))
        out = Image.composite(Image.new('RGB', (w, h), LIP), out,
                              ring_m.filter(ImageFilter.GaussianBlur(1.1)))
        return out

    return {n: pose(*p) for n, p in POSES.items()}


def compose(cells, labels=True):
    w, h = cells['AI'].size
    pad, lab = 14, (52 if labels else 0)
    sheet = Image.new('RGB', (w * 3 + pad * 4, (h + lab) * 3 + pad * 4), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    font = None
    if labels:
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 40)
        except OSError:
            font = ImageFont.load_default()
    for i, n in enumerate(ORDER):
        r, c = divmod(i, 3)
        x = pad + c * (w + pad)
        y = pad + r * (h + lab + pad)
        if labels:
            t = 'etc / rest' if n == 'REST' else n
            dr.text((x + w / 2 - dr.textlength(t, font=font) / 2, y + 4), t,
                    fill=(0, 0, 0), font=font)
        sheet.paste(cells[n], (x, y + lab))
    return sheet


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', help='a single front-facing face')
    p.add_argument('--out', default='sheet.png')
    p.add_argument('--toon-out', help='also write the cartoonified face alone')
    p.add_argument('--scale', type=int, default=5, help='upscale factor (small sources need more)')
    p.add_argument('--sat', type=float, default=1.25)
    p.add_argument('--bg-tol', type=int, default=30, help='background keying tolerance')
    p.add_argument('--open', type=float, default=0.105, help='max jaw drop, fraction of height')
    p.add_argument('--no-labels', action='store_true')
    a = p.parse_args()

    bgr = cv2.imread(a.image)
    if bgr is None:
        raise SystemExit(f'cannot read {a.image}')
    rgb, fg = cartoonify(bgr, a.scale, a.sat, a.bg_tol)
    lip, cx, half = find_mouth(rgb)
    print(f'face {rgb.shape[1]}x{rgb.shape[0]}  lip at {lip/rgb.shape[0]:.0%} down, '
          f'{cx/rgb.shape[1]:.0%} across')
    if a.toon_out:
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(a.toon_out)
    sheet = compose(build(rgb, lip, cx, half, a.open), not a.no_labels)
    sheet.save(a.out)
    print(f'wrote {a.out}  {sheet.size[0]}x{sheet.size[1]}')


if __name__ == '__main__':
    main()
