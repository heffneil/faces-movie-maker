#!/usr/bin/env python3
"""One cartoon face -> a 9-viseme phoneme sheet.

Give it a single flat cartoon portrait (bold outlines, flat colour fills) and
it writes a 3x3 sheet `sprite.py` can animate. The existing mouth is erased by
inpainting and nine mouths are drawn back in the artwork's own style: flat
fills, a bold outline matched to the source linework, teeth and a tongue.

Every cell shares one identical head — only the mouth differs — which is what
makes the animation register cleanly. Nothing warps the face.

The source should already be cartoon art. `--from-photo` runs a cartoonify
prefilter for photographic input, but it is a poor substitute: filters smooth
a photograph, they do not produce flat colour and linework, and mouths drawn
over smooth shading read as pasted on. Cartoonify the face properly first.

Usage:
    .venv/bin/python facesheet.py face.png --out sheet.png
    .venv/bin/python facesheet.py face.png --out sheet.png --preview AI
    .venv/bin/python facesheet.py face.png --out sheet.png --lip 690 --half 0.09
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# open, width, roundness, upper teeth, tongue, lower teeth
POSES = {
    'AI':   (1.00, 0.96, 0.00, 0.36, 0.34, 0.20),
    'E':    (0.46, 1.16, 0.00, 0.74, 0.00, 0.30),
    'FV':   (0.16, 0.98, 0.00, 0.95, 0.00, 0.00),
    'L':    (0.58, 0.84, 0.05, 0.34, 0.80, 0.00),
    'MBP':  (0.00, 1.00, 0.00, 0.00, 0.00, 0.00),
    'O':    (0.74, 0.60, 0.90, 0.10, 0.00, 0.00),
    'U':    (0.52, 0.50, 1.00, 0.00, 0.00, 0.00),
    'WQ':   (0.34, 0.46, 1.00, 0.00, 0.00, 0.00),
    'REST': (0.10, 1.00, 0.00, 0.00, 0.00, 0.00),
}
ORDER = ['AI', 'E', 'FV', 'L', 'MBP', 'O', 'U', 'WQ', 'REST']
SS = 3                                    # supersample factor for crisp edges


def cartoonify(bgr):
    """Prefilter for photographic input. Flattens shading while keeping the
    features; colour quantisation was tried and destroyed glasses and eyes."""
    out = cv2.edgePreservingFilter(bgr, flags=1, sigma_s=64, sigma_r=0.4)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def find_lip(rgb):
    """Lip line = darkest row of the central face strip, nose to chin."""
    h, w = rgb.shape[:2]
    y0, y1 = int(h * 0.64), int(h * 0.82)
    prof = rgb[y0:y1, w // 2 - 70:w // 2 + 70].mean(axis=(1, 2))
    return y0 + int(np.argmin(prof)), w // 2


def palette(rgb, lip, cx):
    flat = rgb.reshape(-1, 3)
    ink = tuple(np.percentile(flat, 0.4, axis=0).astype(int))
    lipc = np.median(rgb[lip - 14:lip - 6, cx - 45:cx + 45].reshape(-1, 3), axis=0)
    return {'ink': ink, 'lip': tuple(np.clip(lipc * 0.97, 0, 255).astype(int)),
            'dark': (74, 32, 38), 'teeth': (252, 247, 238), 'tongue': (204, 100, 110)}


def erase_mouth(bgr, lip, cx):
    """Inpaint the existing mouth away so each pose starts from a clean face."""
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (cx, lip), (int(w * 0.125), int(h * 0.055)), 0, 0, 360, 255, -1)
    return cv2.inpaint(bgr, m, 12, cv2.INPAINT_TELEA)


def draw_mouth(p, size, half, openmax, lw, pal):
    """One mouth as an RGBA patch: flat fills, bold outline, drawn oversized
    and downscaled so the edges anti-alias like the source art."""
    op, wf, rnd, th, tg, bt = p
    bw, bh = size
    W2, H2 = bw * SS, bh * SS
    lay = Image.new('RGBA', (W2, H2), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    ox, oy = W2 // 2, H2 // 2
    mw = half * wf * SS
    mh = max(openmax * op * SS, lw * SS * 0.7)
    lwq = lw * SS
    ink, lipc = pal['ink'] + (255,), pal['lip'] + (255,)

    pad = lwq * 1.5
    d.ellipse([ox - mw - pad * 1.15, oy - mh - pad,
               ox + mw + pad * 1.15, oy + mh + pad], fill=lipc)
    if op < 0.02:                                     # closed: a lip line only
        d.line([ox - mw, oy, ox + mw, oy], fill=ink, width=int(lwq))
        return lay.resize((bw, bh), Image.LANCZOS)

    box = [ox - mw, oy - mh, ox + mw, oy + mh]
    d.ellipse(box, fill=pal['dark'] + (255,))
    inner = Image.new('L', (W2, H2), 0)
    ImageDraw.Draw(inner).ellipse(box, fill=255)

    det = Image.new('RGBA', (W2, H2), (0, 0, 0, 0))
    dd = ImageDraw.Draw(det)
    if tg > 0:
        dd.ellipse([ox - mw * 0.66, oy + mh - mh * 1.55 * tg,
                    ox + mw * 0.66, oy + mh + mh * 0.30], fill=pal['tongue'] + (255,))
    if th > 0:
        dd.ellipse([ox - mw * 0.99, oy - mh - mh * 0.50,
                    ox + mw * 0.99, oy - mh + mh * 1.45 * th], fill=pal['teeth'] + (255,))
    if bt > 0:
        dd.ellipse([ox - mw * 0.92, oy + mh - mh * 1.30 * bt,
                    ox + mw * 0.92, oy + mh + mh * 0.45], fill=pal['teeth'] + (255,))
    lay.paste(det, (0, 0),
              Image.composite(det.split()[3], Image.new('L', (W2, H2), 0), inner))
    d.ellipse(box, outline=ink, width=int(lwq))
    return lay.resize((bw, bh), Image.LANCZOS)


def compose(cells, w, h, labels=True):
    pad, lab = 16, (56 if labels else 0)
    sheet = Image.new('RGB', (w * 3 + pad * 4, (h + lab) * 3 + pad * 4), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    font = None
    if labels:
        try:
            font = ImageFont.truetype(
                '/System/Library/Fonts/Supplemental/Arial Bold.ttf', 42)
        except OSError:
            font = ImageFont.load_default()
    for i, n in enumerate(ORDER):
        r, c = divmod(i, 3)
        x, y = pad + c * (w + pad), pad + r * (h + lab + pad)
        if labels:
            t = 'etc / rest' if n == 'REST' else n
            dr.text((x + w / 2 - dr.textlength(t, font=font) / 2, y + 5), t,
                    fill=(0, 0, 0), font=font)
        sheet.paste(cells[n], (x, y + lab))
    return sheet


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image', help='a flat cartoon portrait, facing forward')
    ap.add_argument('--out', default='sheet.png')
    ap.add_argument('--from-photo', action='store_true',
                    help='cartoonify first (poor substitute for real cartoon art)')
    ap.add_argument('--lip', type=int, help='override the detected lip row')
    ap.add_argument('--cx', type=int, help='override the mouth centre column')
    ap.add_argument('--half', type=float, default=0.082, help='mouth half-width / image width')
    ap.add_argument('--open', type=float, default=0.052, help='max opening half-height / height')
    ap.add_argument('--preview', help='also write a close-up of this pose, e.g. AI')
    ap.add_argument('--no-labels', action='store_true')
    a = ap.parse_args()

    bgr = cv2.imread(a.image)
    if bgr is None:
        raise SystemExit(f'cannot read {a.image}')
    if a.from_photo:
        bgr = cartoonify(bgr)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    lip, cx = find_lip(rgb)
    lip, cx = (a.lip or lip), (a.cx or cx)
    print(f'{w}x{h}  mouth at row {lip} ({lip/h:.0%} down), column {cx} ({cx/w:.0%} across)')

    pal = palette(rgb, lip, cx)
    base = Image.fromarray(cv2.cvtColor(erase_mouth(bgr, lip, cx), cv2.COLOR_BGR2RGB))
    size = (int(w * 0.30), int(h * 0.26))
    lw = max(3, int(w * 0.0052))

    cells = {}
    for n, p in POSES.items():
        cell = base.copy()
        patch = draw_mouth(p, size, w * a.half, h * a.open, lw, pal)
        cell.paste(patch, (cx - size[0] // 2, lip - size[1] // 2), patch)
        cells[n] = cell

    compose(cells, w, h, not a.no_labels).save(a.out)
    print(f'wrote {a.out}')
    if a.preview:
        n = a.preview.upper()
        if n in cells:
            f = os.path.splitext(a.out)[0] + f'-{n}.png'
            cells[n].crop((cx - 260, lip - 210, cx + 260, lip + 210)).save(f)
            print(f'wrote {f}')


if __name__ == '__main__':
    main()
