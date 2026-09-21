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
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# open, width, roundness, upper teeth, tongue, lower teeth, lower-lip boost
#
# Sizes are chosen so the three round shapes stay tellable apart at a glance
# on a display set: O is the big circle, U a distinctly taller-than-wide oval,
# WQ a small tight pucker. FV is the labiodental — upper teeth resting on the
# lower lip — so its teeth nearly fill the gap and its lower lip is thickened,
# otherwise it collapses into something that reads as a closed line.
POSES = {
    'AI':   (1.00, 0.96, 0.00, 0.36, 0.34, 0.20, 0.0),
    'E':    (0.56, 1.16, 0.00, 0.78, 0.00, 0.30, 0.0),
    'FV':   (0.28, 0.98, 0.00, 1.10, 0.00, 0.00, 1.1),
    'L':    (0.58, 0.84, 0.05, 0.34, 0.80, 0.00, 0.0),
    'MBP':  (0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.0),
    'O':    (0.82, 0.64, 0.90, 0.10, 0.00, 0.00, 0.0),
    'U':    (0.54, 0.44, 1.00, 0.00, 0.00, 0.00, 0.0),
    'WQ':   (0.32, 0.30, 1.00, 0.00, 0.00, 0.00, 0.0),
    'REST': (0.10, 1.00, 0.00, 0.00, 0.00, 0.00, 0.0),
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
    lipc = np.median(np.vstack([rgb[lip - 16:lip - 8, cx - 55:cx + 55].reshape(-1, 3),
                                rgb[lip + 12:lip + 22, cx - 55:cx + 55].reshape(-1, 3)]),
                     axis=0)
    # lips in this style barely differ from skin: a thick pale ring reads as a
    # donut round the mouth, so keep it close to the surrounding tone
    lipc = lipc * np.array([0.95, 0.88, 0.87])
    return {'ink': ink, 'lip': tuple(np.clip(lipc, 0, 255).astype(int)),
            'dark': (74, 32, 38), 'teeth': (252, 247, 238), 'tongue': (204, 100, 110)}


def erase_mouth(bgr, lip, cx):
    """Blank the existing mouth so each pose starts from a clean face.

    Inpainting smeared the old lip darkness across the chin and left a visible
    grey streak under every pose. The artwork is flat colour, so filling with
    the surrounding skin tone is both simpler and seamless.
    """
    h, w = bgr.shape[:2]
    ax, ay = int(w * 0.132), int(h * 0.060)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (cx, lip), (ax, ay), 0, 0, 360, 255, -1)
    ring = cv2.dilate(m, np.ones((31, 31), np.uint8)) - m
    skin = np.median(bgr[ring > 0], axis=0)
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 5)[..., None]
    return np.clip(bgr * (1 - soft) + skin * soft, 0, 255).astype(np.uint8)


def _lip_path(mw, mh_up, mh_dn, rnd, bow=0.0, n=96):
    """Points along the upper and lower edges of a mouth.

    An ellipse is blunt at the corners; a real mouth tapers to a point there,
    which is most of why drawn-on ellipses look wrong. `rnd` blends between
    that tapered almond (0) and a true circle (1) for the rounded vowels.
    `bow` dips the centre of the upper edge into a cupid's bow.
    """
    up, dn = [], []
    for i in range(n + 1):
        t = i / n
        s = math.sin(math.pi * t)
        x = (1 - rnd) * (-mw + 2 * mw * t) + rnd * (-mw * math.cos(math.pi * t))
        au = (1 - rnd) * s ** 0.8 + rnd * s
        ad = (1 - rnd) * s ** 0.9 + rnd * s
        dip = 1 - bow * math.exp(-(((t - 0.5) / 0.14) ** 2))
        up.append((x, -mh_up * au * dip))
        dn.append((x, mh_dn * ad))
    return up, dn


def _poly(up, dn, ox, oy):
    return [(ox + x, oy + y) for x, y in up] + [(ox + x, oy + y) for x, y in reversed(dn)]


def draw_mouth(p, size, half, openmax, lw, pal):
    """One mouth as an RGBA patch: curve-built lips and opening, flat fills,
    bold outline. Drawn oversized and downscaled so edges anti-alias like the
    source art."""
    op, wf, rnd, th, tg, bt, llip = p
    bw, bh = size
    W2, H2 = bw * SS, bh * SS
    lay = Image.new('RGBA', (W2, H2), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    ox, oy = W2 // 2, H2 // 2
    mw = half * wf * SS
    mh = max(openmax * op * SS, lw * SS * 0.6)
    lwq = lw * SS
    ink, lipc = pal['ink'] + (255,), pal['lip'] + (255,)

    # lips: the same silhouette, inflated, so lip thickness follows the mouth
    lt = lwq * 0.85
    lup, ldn = _lip_path(mw + lt * 0.5, mh + lt, mh + lt * (1 + llip * 2.4),
                         rnd, bow=0.26)
    d.polygon(_poly(lup, ldn, ox, oy), fill=lipc)
    d.line(_poly(lup, ldn, ox, oy) + [_poly(lup, ldn, ox, oy)[0]],
           fill=tuple(int(c * 0.82) for c in pal['lip']) + (255,), width=int(lwq * 0.5))

    if op < 0.02:                   # closed: a tapered lens reads as lips
        cu, cd = _lip_path(mw, lwq * 0.55, lwq * 0.55, 0.0, bow=0.35)
        d.polygon(_poly(cu, cd, ox, oy), fill=ink)
        return lay.resize((bw, bh), Image.LANCZOS)

    up, dn = _lip_path(mw, mh, mh, rnd, bow=0.18)
    poly = _poly(up, dn, ox, oy)
    d.polygon(poly, fill=pal['dark'] + (255,))
    inner = Image.new('L', (W2, H2), 0)
    ImageDraw.Draw(inner).polygon(poly, fill=255)

    det = Image.new('RGBA', (W2, H2), (0, 0, 0, 0))
    dd = ImageDraw.Draw(det)
    if tg > 0:                                     # tongue hugs the lower edge
        tu = [(x, y - mh * 1.15 * tg) for x, y in dn]
        dd.polygon(_poly(tu, [(x, y + mh * 0.4) for x, y in dn], ox, oy),
                   fill=pal['tongue'] + (255,))
    if th > 0:                                     # teeth follow the upper arc
        td = [(x, y + mh * 1.45 * th) for x, y in up]
        dd.polygon(_poly(up, td, ox, oy), fill=pal['teeth'] + (255,))
    if bt > 0:
        bu = [(x, y - mh * 1.30 * bt) for x, y in dn]
        dd.polygon(_poly(bu, dn, ox, oy), fill=pal['teeth'] + (255,))
    lay.paste(det, (0, 0),
              Image.composite(det.split()[3], Image.new('L', (W2, H2), 0), inner))

    if th > 1.0:      # labiodental: a shadow keeps the teeth off the lower lip
        sd = Image.new('RGBA', (W2, H2), (0, 0, 0, 0))
        edge = [(x, y + mh * 1.45 * th) for x, y in up]
        ImageDraw.Draw(sd).line(_poly(edge, edge, ox, oy),
                                fill=pal['dark'] + (255,), width=int(lwq * 1.1))
        lay.paste(sd, (0, 0),
                  Image.composite(sd.split()[3], Image.new('L', (W2, H2), 0), inner))

    d.line(poly + [poly[0]], fill=ink, width=int(lwq), joint='curve')
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
