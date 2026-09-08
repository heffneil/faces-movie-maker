#!/usr/bin/env python3
"""Text -> talking face MP4 using a 3x3 viseme sprite sheet (user-drawn faces).

Sheet layout (black on white, labels above each cell):
    AI   E    FV
    L    MBP  O
    U    WQ   etc/rest
All cells share identical eyes/nose; only the mouth differs. Sprites are
aligned by their content top (the eyes), colorized into a glowing projection
face on black, and cut per phoneme.

Usage:
    .venv/bin/python sprite.py "This is a test" --sheet phoenemes.jpeg --out x.mp4
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

from pumpkin import STYLES, envelope, tts, load_audio, FPS

SPRITE_NAMES = ["AI", "E", "FV", "L", "MBP", "O", "U", "WQ", "REST"]

PH2SPRITE = {
    **{c: "AI" for c in "ɑaæʌɐAIh"},
    **{c: "E" for c in "iɪɛeəj"},
    **{c: "FV" for c in "fv"},
    "l": "L",
    **{c: "MBP" for c in "mbp"},
    **{c: "O" for c in "oɔO"},
    **{c: "U" for c in "uʊU"},
    **{c: "WQ" for c in "wWQ"},
}
SKIP = set("ˈˌː ̩ᵊ ")


def slice_sheet(path, label_trim=0.16):
    """Return dict name -> binary mask (bool array), aligned to common canvas."""
    from PIL import Image

    sheet = Image.open(path).convert("L")
    w, h = sheet.size
    caption = int(h * 0.06)                      # bottom caption strip
    cw, ch = w // 3, (h - caption) // 3
    masks = {}
    for idx, name in enumerate(SPRITE_NAMES):
        r, c = divmod(idx, 3)
        cell = sheet.crop((c * cw, r * ch + int(ch * label_trim),
                           (c + 1) * cw, (r + 1) * ch))
        a = np.array(cell) < 110                 # black ink -> True
        # drop small blobs (the text labels); face parts are far larger
        from scipy import ndimage
        lab, n = ndimage.label(a)
        sizes = ndimage.sum(a, lab, range(1, n + 1))
        cy = ndimage.center_of_mass(a, lab, range(1, n + 1))
        ok = [i + 1 for i in range(n)
              if sizes[i] > a.size * 0.004
              and not (sizes[i] < a.size * 0.02 and cy[i][0] < a.shape[0] * 0.3)]
        masks[name] = np.isin(lab, ok)
    # align: identical eyes -> align by content bbox top + horizontal center
    crops = {}
    for name, m in masks.items():
        ys, xs = np.where(m)
        crops[name] = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    canvases = {}
    H = max(c.shape[0] for c in crops.values()) + 40
    W = max(max(c.shape[1] for c in crops.values()) + 40, cw)
    for name, crop in crops.items():
        canvas = np.zeros((H, W), dtype=bool)
        x0 = (W - crop.shape[1]) // 2
        canvas[20:20 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
        canvases[name] = canvas
    return canvases


def eye_boxes(mask):
    """Bounding boxes of the two eyes: the largest blobs in the top half."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(lab):
        y, x = sl
        if y.stop < mask.shape[0] * 0.55:        # top-half component
            boxes.append((y.start, y.stop, x.start, x.stop,
                          (y.stop - y.start) * (x.stop - x.start)))
    boxes.sort(key=lambda b: -b[4])
    return [b[:4] for b in boxes[:2]]


def apply_blink(field, boxes, f):
    """Squash the eye regions vertically by factor f (1=open, ~0.08=shut)."""
    out = field.copy()
    far = np.float32(24.0)
    for y0, y1, x0, x1 in boxes:
        cy = (y0 + y1) / 2
        ys = np.arange(y0, y1)
        src = np.clip((cy + (ys - cy) / max(f, 0.02)), y0, y1 - 1).astype(int)
        band = field[src, x0:x1].copy()
        unmapped = (ys - cy) / max(f, 0.02) + cy
        band[(unmapped >= y1 - 1) | (unmapped <= y0), :] = far
        out[y0:y1, x0:x1] = band
    return out


def slice_sheet_color(path, label_trim=0.16):
    """Full-color 3x3 sheet -> dict name -> premultiplied RGBA float32 array.
    Background = near-white pixels connected to the border (a white beard
    inside the face survives). Labels are dropped as small stray blobs."""
    from PIL import Image
    from scipy import ndimage

    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    ink = img.min(axis=2) < 235
    # find the 9 face blobs anywhere on the sheet (ignores grid lines/clipping)
    grown = ndimage.binary_closing(ink, iterations=3)
    lab, n = ndimage.label(grown)
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        size = (lab[sl] == i).sum()
        comps.append((size, i, sl))
    comps.sort(reverse=True)
    comps = comps[:9]
    if len(comps) < 9:
        raise ValueError(f"found only {len(comps)} face blobs, need 9")
    # order into 3 rows of 3 by position
    items = []
    for size, i, (ys, xs) in comps:
        items.append(((ys.start + ys.stop) / 2, (xs.start + xs.stop) / 2, i, (ys, xs)))
    items.sort()
    rows = [sorted(items[k:k + 3], key=lambda t: t[1]) for k in (0, 3, 6)]
    cells = {}
    for idx, name in enumerate(SPRITE_NAMES):
        _, _, ci, (ys, xs) = rows[idx // 3][idx % 3]
        pad = 6
        y0, y1 = max(0, ys.start - pad), min(img.shape[0], ys.stop + pad)
        x0, x1 = max(0, xs.start - pad), min(img.shape[1], xs.stop + pad)
        cell = img[y0:y1, x0:x1]
        # re-isolate in raw ink so label text bridged by the closing step drops
        raw = ink[y0:y1, x0:x1] & (lab[y0:y1, x0:x1] == ci)
        lab3, n3 = ndimage.label(raw)
        if n3 > 1:
            sizes3 = ndimage.sum(raw, lab3, range(1, n3 + 1))
            keep = int(np.argmax(sizes3)) + 1
            # keep the face plus any sizable satellite (a detached pom-pom),
            # but drop small text blobs
            mask = np.isin(lab3, [k + 1 for k in range(n3)
                                  if k + 1 == keep or sizes3[k] > sizes3.max() * 0.05])
        else:
            mask = lab3 > 0
        # fill interior: white areas inside the outline (a white beard on a
        # white page) become part of the silhouette
        fg = ndimage.binary_fill_holes(mask)
        alpha = ndimage.gaussian_filter(fg.astype(np.float32), 1.0)
        cells[name] = (cell * alpha[..., None] / 255.0, alpha)
    # rough placement on a common canvas by content bbox center
    boxes = {}
    for name, (rgb, a) in cells.items():
        ys, xs = np.where(a > 0.5)
        boxes[name] = (ys.min(), ys.max(), xs.min(), xs.max())
    H = max(b[1] - b[0] for b in boxes.values()) + 81
    W = max(b[3] - b[2] for b in boxes.values()) + 81
    out = {}
    for name, (rgb, a) in cells.items():
        y0, y1, x0, x1 = boxes[name]
        rgba = np.zeros((H, W, 4), np.float32)
        oy = (H - (y1 - y0 + 1)) // 2
        ox = (W - (x1 - x0 + 1)) // 2
        rgba[oy:oy + y1 - y0 + 1, ox:ox + x1 - x0 + 1, :3] = rgb[y0:y1 + 1, x0:x1 + 1]
        rgba[oy:oy + y1 - y0 + 1, ox:ox + x1 - x0 + 1, 3] = a[y0:y1 + 1, x0:x1 + 1]
        out[name] = rgba
    # refine: register each sprite to REST on the head region (top 55%) so
    # hard pose cuts don't make the head jitter (mouth changes move bboxes)
    ref = out["REST"][:int(H * 0.55), :, 3][::2, ::2]
    for name in SPRITE_NAMES:
        if name == "REST":
            continue
        a = out[name][:int(H * 0.55), :, 3][::2, ::2]
        best, bdy, bdx = -1.0, 0, 0
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                s = (np.roll(np.roll(a, dy, 0), dx, 1) * ref).sum()
                if s > best:
                    best, bdy, bdx = s, dy, dx
        out[name] = np.roll(np.roll(out[name], bdy * 2, axis=0), bdx * 2, axis=1)
    return out


def drift(frame, wob):
    """Continuous sub-pixel drift + slight rotation so every frame moves
    smoothly (integer-pixel wobble makes held poses look frozen/steppy)."""
    from PIL import Image
    W, H = frame.size
    dx = 5.0 * math.sin(wob * 1.3)
    dy = 4.0 * math.sin(wob * 0.9 + 2)
    ang = 0.010 * math.sin(wob * 0.63)          # ~0.6 degrees
    cos, sin = math.cos(ang), math.sin(ang)
    cx, cy = W / 2 + dx, H / 2 + dy
    coeffs = (cos, sin, -cos * cx - sin * cy + W / 2,
              -sin, cos, sin * cx - cos * cy + H / 2)
    return frame.transform((W, H), Image.AFFINE, coeffs, resample=Image.BILINEAR)


class RegionMorpher:
    """True in-betweens for flat cartoon art.

    The mouth region, the colour palette and the per-pose shape decomposition
    are all computed ONCE for the whole sheet, not per pose pair. That matters:
    a per-pair palette or region silently shifts the painted colours and the
    painted area every time the morph target changes — which in running speech
    is ~12 times a second, and reads as flicker.

    Each pose is stored as k flat-colour layers described by signed distance
    fields; morphing interpolates those fields and repaints, so intermediate
    frames are real mouth shapes rather than pixel blends.
    """

    def __init__(self, sprites, k=5, seed=7):
        from scipy import ndimage

        self.sprites = {n: np.clip(s, 0, 1).astype(np.float32)
                        for n, s in sprites.items()}
        self.names = list(self.sprites)
        self.index = {n: i for i, n in enumerate(self.names)}
        rng = np.random.RandomState(seed)

        stack = np.stack([self.sprites[n][..., :3] for n in self.names])

        # --- one region for every pair: where any pose differs from the mean
        var = stack.std(axis=0).max(axis=-1)
        m = ndimage.binary_opening(var > 0.15, iterations=2)
        if not m.any():
            self.box = None
            return
        lab, nlab = ndimage.label(ndimage.binary_dilation(m, iterations=5))
        sizes = ndimage.sum(m, lab, range(1, nlab + 1))
        keep = lab == int(np.argmax(sizes)) + 1
        soft = ndimage.gaussian_filter(keep.astype(np.float32), 3)
        ys, xs = np.where(keep)
        pad = 8
        y0, y1 = max(0, ys.min() - pad), min(var.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(var.shape[1], xs.max() + pad)
        self.box = (y0, y1, x0, x1)
        self.soft = soft[y0:y1, x0:x1][..., None]
        region = stack[:, y0:y1, x0:x1]

        # --- one palette shared by every pose, so colours never shift
        px = region.reshape(-1, 3)
        sample = px[rng.choice(len(px), min(30000, len(px)), replace=False)]
        cent = sample[rng.choice(len(sample), k, replace=False)].copy()
        for _ in range(15):
            lb = ((sample[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
            for c in range(k):
                if (lb == c).any():
                    cent[c] = sample[lb == c].mean(0)
        self.colors = cent
        self.k = k
        self.dark = int(np.argmin(cent.sum(-1)))       # mouth interior cluster

        # --- per pose: a distance field per colour layer, plus interior area
        self.fields, self.area, counts = {}, {}, np.zeros(k)
        for i, name in enumerate(self.names):
            lb = ((region[i][..., None, :] - cent[None, None]) ** 2).sum(-1).argmin(-1)
            fs = []
            for c in range(k):
                mask = lb == c
                counts[c] += mask.sum()
                if mask.any() and (~mask).any():
                    fs.append(np.clip(ndimage.distance_transform_edt(~mask)
                                      - ndimage.distance_transform_edt(mask),
                                      -20, 20).astype(np.float32))
                else:
                    fs.append(np.full(mask.shape, 20.0 if not mask.any() else -20.0,
                                      np.float32))
            self.fields[name] = fs
            self.area[name] = float((lb == self.dark).sum())
        self.order = np.argsort(-counts)               # paint big layers first

        h, wd = region.shape[1:3]
        self.yn = (np.linspace(0, 1, h, dtype=np.float32)[:, None]
                   * np.ones((1, wd), np.float32))
        self.edge = (np.ones((h, 1), np.float32)
                     * np.abs(np.linspace(0, 1, wd, dtype=np.float32) - 0.5)[None, :] * 2.0)

    def frame(self, a, b, w, asym=True):
        """Premultiplied RGBA canvas of pose a->b at weight w.

        The pair is ordered canonically, so frame(a, b, w) and frame(b, a, 1-w)
        return the identical image — without that, every crossover between two
        poses would pop.
        """
        if self.index[a] > self.index[b]:
            a, b, w = b, a, 1.0 - w
        out = (1 - w) * self.sprites[a] + w * self.sprites[b]
        if self.box is None or w <= 0.0:
            return out
        y0, y1, x0, x1 = self.box

        if asym:
            # A mouth does not close uniformly: the lower lip travels further
            # than the upper, and the corners meet before the centre does
            # (opening runs the other way). The bias is scaled by 4w(1-w), which
            # vanishes at w=0 and w=1, so both ends land exactly on their pose.
            closing = 1.0 if self.area[a] > self.area[b] else -1.0
            bell = 4.0 * w * (1.0 - w)
            wf = np.clip(w + bell * (0.30 * (self.yn - 0.5)
                                     + 0.24 * closing * (self.edge - 0.5)),
                         0.0, 1.0).astype(np.float32)
        else:
            wf = np.float32(w)

        fa, fb = self.fields[a], self.fields[b]
        paint = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        for c in self.order:
            f = (1 - wf) * fa[c] + wf * fb[c]
            alpha = np.clip(0.5 - f / 1.5, 0, 1)[..., None]
            paint = paint * (1 - alpha) + self.colors[c] * alpha
        reg = out[y0:y1, x0:x1, :3]
        out[y0:y1, x0:x1, :3] = reg * (1 - self.soft) + paint * self.soft
        return out


def compose_color(state, W, H, wob, stretch):
    """Premultiplied RGBA float array -> full frame over black."""
    from PIL import Image

    sh, sw = state.shape[:2]
    scale = min(W * 0.8 / sw, H * 0.8 / sh)
    tw = int(sw * scale * (1 - 0.4 * stretch))
    th = int(sh * scale * (1 + stretch))
    img = Image.fromarray((np.clip(state[..., :3], 0, 1) * 255).astype(np.uint8))
    img = img.resize((tw, th), Image.LANCZOS)  # premultiplied: over black as-is
    frame = Image.new("RGB", (W, H), (0, 0, 0))
    dx = (W - tw) // 2
    dy = (H - th) // 2
    frame.paste(img, (dx, dy))
    return frame


def sprite_track(tokens, env, n_frames, fps=FPS):
    """Per-frame sprite name, phoneme-timed, min 2-frame hold."""
    track = ["REST"] * n_frames
    for start, end, phonemes in tokens:
        ph = [c for c in phonemes if c not in SKIP]
        if not ph:
            continue
        dur = (end - start) / len(ph)
        for k, c in enumerate(ph):
            f0 = int((start + k * dur) * fps)
            f1 = max(f0 + 1, int((start + (k + 1) * dur) * fps))
            for f in range(f0, min(f1, n_frames)):
                track[f] = PH2SPRITE.get(c, "REST")
    for i, e in enumerate(env):                  # silence -> rest face
        if e == 0:
            track[i] = "REST"
    for i in range(1, n_frames - 1):             # kill 1-frame flickers
        if track[i] != track[i - 1] and track[i] != track[i + 1]:
            track[i] = track[i - 1]
    return track


# How much a viewer notices a viseme. Lip closure on m/b/p and the big open
# vowels are the shapes an audience reads; a neutral consonant rest is filler.
SALIENCE = {"MBP": 3, "AI": 3, "O": 3, "U": 3,
            "E": 2, "L": 2, "FV": 2, "WQ": 2, "REST": 1}


def _segments(track):
    segs, s = [], 0
    for i in range(1, len(track) + 1):
        if i == len(track) or track[i] != track[s]:
            segs.append([s, i, track[s]])
            s = i
    return segs


def enforce_min_hold(track, fps=FPS, min_ms=110):
    """Absorb visemes too brief to read, the way a lip-sync artist would.

    Raw phoneme timings hand us a new mouth shape every two or three frames —
    measured on ordinary TTS speech, two thirds of segments last under 130 ms.
    Animating each one is both unreadable (the mouth never arrives anywhere)
    and jittery. Papagayo and friends solve this with a minimum hold, and so do
    we: a segment shorter than `min_ms` is either grown at the expense of a
    neighbour, when it is the more salient shape (a lip closure on "m" is worth
    protecting), or absorbed into the neighbour that reads more strongly.
    """
    min_f = max(2, int(round(min_ms / 1000 * fps)))
    segs = _segments(track)
    for _ in range(len(segs) * 4):                 # always terminates; bounded
        short = [i for i, (s0, s1, _) in enumerate(segs) if s1 - s0 < min_f]
        if not short or len(segs) < 2:
            break
        i = min(short, key=lambda j: segs[j][1] - segs[j][0])
        s0, s1, name = segs[i]
        left = segs[i - 1] if i > 0 else None
        right = segs[i + 1] if i + 1 < len(segs) else None
        cands = [c for c in (left, right) if c is not None]
        best = max(cands, key=lambda c: (SALIENCE.get(c[2], 2), c[1] - c[0]))

        # more salient than both neighbours? grow it instead of losing it
        donor = max((c for c in cands if c[1] - c[0] > min_f),
                    key=lambda c: c[1] - c[0], default=None)
        if (SALIENCE.get(name, 2) > SALIENCE.get(best[2], 2)) and donor is not None:
            need = min(min_f - (s1 - s0), (donor[1] - donor[0]) - min_f)
            if need > 0:
                if donor is left:
                    donor[1] -= need
                    segs[i][0] -= need
                else:
                    donor[0] += need
                    segs[i][1] += need
                continue

        # otherwise fold it into whichever neighbour reads more strongly
        if best is left:
            left[1] = s1
        else:
            right[0] = s0
        segs.pop(i)

    out = list(track)
    for s0, s1, name in segs:
        for f in range(s0, s1):
            out[f] = name
    return out


def coarticulate(track, fps=FPS, co_ms=28, min_hold_ms=150):
    """Per-frame (pose_a, pose_b, w) from a per-frame pose track.

    The pose of the segment a frame belongs to always dominates, so the mouth
    lands on the sound being made. Blending is confined to the boundaries: `w`
    is 0.5 exactly at a segment edge and decays toward 0 across roughly `co_ms`
    into the segment, which gives

      - anticipation and follow-through, since a frame near a boundary is
        already part way toward the neighbouring pose;
      - undershoot, because a segment shorter than the blend window never gets
        far enough from its edges for w to reach 0 — exactly what a real mouth
        does in fast speech;
      - landing, because the middle of any segment longer than that window
        reaches its pure pose and holds it.

    The partner is always the adjacent segment in time, never whichever pose
    happens to score highest globally: an unstable partner makes the morph
    target flicker. The partner switches at the middle of a segment, where w is
    at its smallest and the change is invisible.

    `co_ms` is in milliseconds, so the animation feels the same at any frame
    rate. Crossovers are continuous: the last frame of one segment and the
    first of the next both sit at w=0.5 on the same pose pair, and
    RegionMorpher orders pairs canonically so they render identically.
    """
    n = len(track)
    if n == 0:
        return []
    track = enforce_min_hold(track, fps, min_hold_ms)
    segs = [tuple(s) for s in _segments(track)]

    sigma = max(0.6, co_ms / 1000 * fps)
    out = []
    for k, (s0, s1, name) in enumerate(segs):
        prev = segs[k - 1][2] if k > 0 else None
        nxt = segs[k + 1][2] if k + 1 < len(segs) else None
        for f in range(s0, s1):
            d_prev, d_next = f - s0, (s1 - 1) - f
            if d_prev <= d_next:
                partner, d = prev, d_prev
            else:
                partner, d = nxt, d_next
            if partner is None:
                out.append((name, name, 0.0))
            else:
                out.append((name, partner, 0.5 * math.exp(-0.5 * (d / sigma) ** 2)))
    return out


def sdf(mask):
    """Signed distance field: negative inside the shape, positive outside."""
    from scipy import ndimage
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    field = (outside - inside).astype(np.float32)
    # light blur irons out JPEG-artifact raggedness in the source edges
    return ndimage.gaussian_filter(field, 1.5)


def colorize(field, style, W, H, wob, stretch=0.0):
    """SDF array -> glowing face frame. The field is rescaled (not the bitmap)
    and re-thresholded at output size, so edges stay crisp at any resolution."""
    from PIL import Image, ImageFilter

    c = STYLES[style]
    sh, sw = field.shape
    scale = min(W * 0.8 / sw, H * 0.8 / sh)
    tw = int(sw * scale * (1 - 0.4 * stretch))
    th = int(sh * scale * (1 + stretch))
    field = np.clip(field, -24.0, 24.0)  # bound jumps so bicubic doesn't ring
    big = Image.fromarray(field, mode="F").resize((tw, th), Image.BICUBIC)
    # distance units are source pixels; feather ~2 output pixels
    alpha = np.clip(0.5 - np.asarray(big) * scale / 2.0, 0.0, 1.0)
    m = Image.fromarray((alpha * 255).astype(np.uint8))
    dx = (W - m.width) // 2
    dy = (H - m.height) // 2
    full = Image.new("L", (W, H), 0)
    full.paste(m, (dx, dy))

    frame = Image.new("RGB", (W, H), (0, 0, 0))
    glow = Image.composite(Image.new("RGB", (W, H), c["glow"]), frame, full)
    frame = glow.filter(ImageFilter.GaussianBlur(int(min(W, H) * 0.04)))
    halo = Image.composite(Image.new("RGB", (W, H), c["mid"]), Image.new("RGB", (W, H), (0, 0, 0)), full)
    frame = Image.blend(frame, halo.filter(ImageFilter.GaussianBlur(int(min(W, H) * 0.012))), 0.6)
    core = Image.new("RGB", (W, H), c["core"])
    frame = Image.composite(core, frame, full)  # soft alpha = anti-aliased edges
    return frame


def _spread_words(text, start, end, g2p):
    """Phonemize `text` and spread its words across [start, end] seconds,
    weighted by phoneme count. Returns (start, end, phonemes) tuples."""
    words = []
    for r in g2p(text):
        for t in (getattr(r, "tokens", None) or []):
            if t.phonemes:
                words.append(t.phonemes)
    if not words:
        return []
    weights = np.array([max(1, len([c for c in w if c not in SKIP])) for w in words], float)
    edges = start + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * (end - start)
    return list(zip(edges[:-1], edges[1:], words))


# xLights/Papagayo timing labels -> our sprites ('etc' is the consonant rest)
XT_VIS = {n: n for n in SPRITE_NAMES} | {"ETC": "REST", "REST": "REST"}


def xtiming_track(xml_text, env, n_frames, fps=FPS):
    """Parse an xLights .xtiming file into a per-frame sprite track.
    Prefers a phoneme layer (AI/E/FV/L/MBP/O/U/WQ/etc labels, frame-accurate);
    falls back to phonemizing a words/phrases layer within its timings."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    layers = []
    for el in root.iter("EffectLayer"):
        effs = []
        for e in el.iter("Effect"):
            label = (e.get("label") or "").strip()
            s = e.get("startTime") or e.get("starttime")   # attr case varies by version
            t = e.get("endTime") or e.get("endtime")
            if label and s and t:
                effs.append((int(s), int(t), label))
        if effs:
            layers.append(effs)
    if not layers:
        raise RuntimeError("no labeled timing effects found in the xtiming file")

    def vis_share(effs):
        return sum(1 for _, _, l in effs if l.upper() in XT_VIS) / len(effs)

    phoneme_layers = [L for L in layers if vis_share(L) > 0.8]
    if phoneme_layers:  # authoritative: use it directly
        track = ["REST"] * n_frames
        for s, e, label in max(phoneme_layers, key=len):
            name = XT_VIS.get(label.upper())
            if not name:
                continue
            f0 = int(s / 1000 * fps)
            f1 = max(f0 + 1, int(e / 1000 * fps))
            for f in range(f0, min(f1, n_frames)):
                track[f] = name
        return track

    # words (or phrases) layer: phonemize each label inside its window
    from kokoro import KPipeline
    g2p = KPipeline(lang_code="a", model=False)
    effs = max(layers, key=len)          # densest layer = words
    tokens = []
    for s, e, label in effs:
        tokens.extend(_spread_words(label, s / 1000, e / 1000, g2p))
    if not tokens:
        raise RuntimeError("could not phonemize any labels in the xtiming file")
    return sprite_track(tokens, env, n_frames, fps)


def aligned_word_tokens(words):
    """Aligner output [{label,start_ms,end_ms}] -> (start, end, phonemes) tuples."""
    from kokoro import KPipeline
    g2p = KPipeline(lang_code="a", model=False)
    tokens = []
    for w in words:
        tokens.extend(_spread_words(w["label"], w["start_ms"] / 1000,
                                    w["end_ms"] / 1000, g2p))
    return tokens


def words_to_xtiming(words, name="lyrics"):
    """Aligner word timings -> xLights .xtiming XML (words + phoneme layers)."""
    from xml.sax.saxutils import quoteattr
    from kokoro import KPipeline
    g2p = KPipeline(lang_code="a", model=False)

    word_fx, ph_fx = [], []
    for w in words:
        s, e = int(w["start_ms"]), int(w["end_ms"])
        word_fx.append((s, e, w["label"]))
        for t0, t1, phonemes in _spread_words(w["label"], s / 1000, e / 1000, g2p):
            # collapse each word's phonemes into Papagayo viseme segments
            ph = [c for c in phonemes if c not in SKIP]
            if not ph:
                continue
            dur = (t1 - t0) / len(ph)
            segs = []
            for k, c in enumerate(ph):
                lab = PH2SPRITE.get(c, "REST")
                lab = "etc" if lab == "REST" else lab
                if segs and segs[-1][2] == lab:
                    segs[-1] = (segs[-1][0], t0 + (k + 1) * dur, lab)
                else:
                    segs.append((t0 + k * dur, t0 + (k + 1) * dur, lab))
            ph_fx.extend((int(a * 1000), int(b * 1000), lab) for a, b, lab in segs)

    def layer(effects):
        rows = "".join(
            f'      <Effect label={quoteattr(l)} starttime="{s}" endtime="{e}"/>\n'
            for s, e, l in effects)
        return f"   <EffectLayer>\n{rows}   </EffectLayer>\n"

    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<timing name={quoteattr(name)} SourceVersion="faces-movie-maker">\n'
            f"{layer(word_fx)}{layer(ph_fx)}</timing>\n")


def lyric_tokens(lrc_text, total_dur):
    """Parse '[mm:ss.xx] lyric line' text -> (start, end, phonemes) word tuples.
    Words in a line share its time window, weighted by phoneme count."""
    import re
    from kokoro import KPipeline

    g2p = KPipeline(lang_code="a", model=False)   # G2P only, no TTS model
    lines = []
    for raw in lrc_text.splitlines():
        m = re.match(r"\s*\[(\d+):(\d+(?:\.\d+)?)\]\s*(.+)", raw)
        if m:
            lines.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    lines.sort(key=lambda x: x[0])
    tokens = []
    for k, (start, text) in enumerate(lines):
        end = lines[k + 1][0] if k + 1 < len(lines) else min(start + 8, total_dur)
        end = min(end, total_dur)
        words = []
        for r in g2p(text):
            for t in (getattr(r, "tokens", None) or []):
                if t.phonemes:
                    words.append(t.phonemes)
        if not words:
            continue
        weights = np.array([max(1, len([c for c in w if c not in SKIP])) for w in words], float)
        # leave a breath gap at the end of each line
        span = (end - start) * 0.92
        edges = start + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * span
        for w, t0, t1 in zip(words, edges[:-1], edges[1:]):
            tokens.append((t0, t1, w))
    return tokens


def render_video(sheet_path, track, env, wav_path, out_path, style="pumpkin",
                 W=1280, H=720, progress=None, fps=FPS):
    """Render a sprite-track to MP4. `progress(done, total)` is optional.
    style 'color' keeps the sheet's own artwork; other styles threshold to ink
    and render as a glowing carve. Both morph shapes, never pixel-blend."""
    if style == "color":
        return render_video_color(sheet_path, track, env, wav_path, out_path,
                                  W, H, progress, fps)
    sprites = slice_sheet(sheet_path)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-i", wav_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         os.path.abspath(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # signed-distance fields per sprite; blending SDFs = smooth shape morphs
    fields = {name: sdf(m) for name, m in sprites.items()}
    blend = coarticulate(track, fps)
    # wide smoothing: the squash follows the shape of a phrase, not every
    # syllable — a fast-reacting scale on a 1080p face reads as pulsing
    e_smooth = np.convolve(env, np.ones(11) / 11, mode="same")

    # random blinks: eased shut-and-open profile, every ~2-5 s
    boxes = eye_boxes(sprites["REST"])
    profile = [0.55, 0.12, 0.06, 0.12, 0.55, 0.85]   # eye openness during a blink
    blink_f = np.ones(len(track))
    rng = np.random.RandomState()
    t = rng.randint(fps, 3 * fps)
    while t < len(track) - len(profile):
        blink_f[t:t + len(profile)] = profile
        t += rng.randint(2 * fps, 5 * fps)

    for i, (a, b, w) in enumerate(blend):
        wob = i / fps * 2 * math.pi * 0.7
        state = fields[a] if w < 0.01 else (1 - w) * fields[a] + w * fields[b]
        shown = apply_blink(state, boxes, blink_f[i]) if blink_f[i] < 1 else state
        stretch = 0.035 * e_smooth[i] + 0.010 * math.sin(wob * 1.1)
        frame = drift(colorize(shown, style, W, H, wob, stretch), wob)
        proc.stdin.write(frame.tobytes())
        if progress:
            progress(i + 1, len(track))
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")


def render_video_color(sheet_path, track, env, wav_path, out_path,
                       W=1280, H=720, progress=None, fps=FPS):
    sprites = slice_sheet_color(sheet_path)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-i", wav_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         os.path.abspath(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wide smoothing: the squash follows the shape of a phrase, not every
    # syllable — a fast-reacting scale on a 1080p face reads as pulsing
    e_smooth = np.convolve(env, np.ones(11) / 11, mode="same")
    # continuous co-articulated motion: every frame is a shape morph between
    # the two strongest pose targets, so the mouth is always travelling and
    # never sits on a blend of two mouths
    morpher = RegionMorpher(sprites)
    blend = coarticulate(track, fps)
    for i, (a, b, w) in enumerate(blend):
        wob = i / fps * 2 * math.pi * 0.7
        state = sprites[a] if w < 0.01 else morpher.frame(a, b, w)
        stretch = 0.035 * e_smooth[i] + 0.010 * math.sin(wob * 1.1)
        frame = drift(compose_color(state, W, H, wob, stretch), wob)
        proc.stdin.write(frame.tobytes())
        if progress:
            progress(i + 1, len(track))
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text")
    p.add_argument("--sheet", required=True, help="3x3 phoneme sheet image")
    p.add_argument("--out", default="sprite.mp4")
    p.add_argument("--style", default="pumpkin", choices=list(STYLES) + ["color"])
    p.add_argument("--voice", default="am_michael")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=int, default=FPS, help="30 (default) or 60")
    args = p.parse_args()
    W, H = (int(v) for v in args.size.split("x"))

    wav = tempfile.mktemp(suffix=".wav")
    print("[1/3] TTS ...")
    _, tokens = tts(args.text, args.voice, wav)
    audio, sr = load_audio(wav)
    n_frames = max(1, int(len(audio) / sr * args.fps))
    env = envelope(audio, sr, n_frames)
    track = sprite_track(tokens, env, n_frames, args.fps)
    print("[2/3] rendering", n_frames, "frames ...")
    render_video(args.sheet, track, env, wav, args.out, args.style, W, H,
                 fps=args.fps)
    os.unlink(wav)
    print(f"[3/3] done: {args.out}")


if __name__ == "__main__":
    main()
