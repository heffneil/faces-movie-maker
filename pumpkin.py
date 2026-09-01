#!/usr/bin/env python3
"""Text -> glowing Halloween projection face MP4 (Trick Ghastly style).

Draws a carved jack-o'-lantern / ghost face on pure black — for projecting
onto pumpkins or windows — with phoneme-shaped mouthing, emoting eyes, and
organic wobble.

Usage:
    .venv/bin/python pumpkin.py "THIS IS A TEST" --out test.mp4
    .venv/bin/python pumpkin.py "..." --style ghost --voice af_heart
    .venv/bin/python pumpkin.py --audio vocals.wav --out sing.mp4
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

FPS = 30

STYLES = {
    "pumpkin": {"core": (255, 205, 90), "mid": (255, 140, 20), "glow": (255, 90, 0)},
    "ghost":   {"core": (255, 255, 255), "mid": (210, 220, 245), "glow": (120, 150, 220)},
}

# phoneme -> (open, width, smile, pucker)
#   open: jaw 0..1   width: mouth span   smile: corner lift -1..1   pucker: 0..1 oval
VIS = {
    **{c: (0.04, 0.90, 0.5, 0.0) for c in "mbp"},              # lips closed
    **{c: (0.12, 0.85, 0.3, 0.0) for c in "fv"},
    **{c: (0.22, 0.95, 0.5, 0.0) for c in "szʃʒʧʤθðtdnlkg"},   # teeth close
    **{c: (1.00, 0.95, 0.2, 0.1) for c in "ɑaæʌɐA"},           # ah — tall open
    **{c: (0.55, 1.00, 0.5, 0.0) for c in "ɛeə"},              # eh
    **{c: (0.35, 1.15, 0.8, 0.0) for c in "ɪiI"},              # ee — wide grin
    **{c: (0.60, 0.50, 0.0, 1.0) for c in "ʊuUoɔOQ"},          # oo — round pucker
    **{c: (0.45, 0.55, 0.1, 0.8) for c in "wW"},
    "ɹ": (0.35, 0.75, 0.3, 0.3), "r": (0.35, 0.75, 0.3, 0.3),
    "j": (0.35, 1.05, 0.6, 0.0), "h": (0.45, 0.90, 0.3, 0.0),
}
VIS_DEFAULT = (0.28, 0.90, 0.4, 0.0)
VIS_REST = np.array((0.05, 0.90, 0.55, 0.0))
SKIP = set("ˈˌː ̩ᵊ ")


def tts(text, voice, wav_path):
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=voice[0])
    chunks, tokens, offset = [], [], 0.0
    for r in pipeline(text, voice=voice):
        chunks.append(r.audio)
        for t in (getattr(r, "tokens", None) or []):
            if t.start_ts is not None and t.phonemes:
                tokens.append((offset + t.start_ts, offset + t.end_ts, t.phonemes))
        offset += len(r.audio) / 24000
    audio = np.concatenate(chunks)
    sf.write(wav_path, audio, 24000)
    return audio, tokens


def load_audio(path):
    import soundfile as sf
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def envelope(audio, sr, n_frames):
    hop = len(audio) / n_frames
    env = np.array([
        np.sqrt(np.mean(audio[int(i * hop):max(int(i * hop) + 1, int((i + 1) * hop))] ** 2))
        for i in range(n_frames)
    ])
    env = np.convolve(env, np.ones(3) / 3, mode="same")
    peak = env.max() or 1.0
    env = env / peak
    env[env < 0.08] = 0.0
    return env


def mouth_track(tokens, env, n_frames):
    """Per-frame (open, width, smile, pucker), phoneme-timed and smoothed."""
    track = np.tile(VIS_REST, (n_frames, 1))
    for start, end, phonemes in tokens:
        ph = [c for c in phonemes if c not in SKIP]
        if not ph:
            continue
        dur = (end - start) / len(ph)
        for k, c in enumerate(ph):
            f0 = int((start + k * dur) * FPS)
            f1 = max(f0 + 1, int((start + (k + 1) * dur) * FPS))
            track[f0:f1] = VIS.get(c, VIS_DEFAULT)
    for i in range(1, n_frames):
        track[i] = 0.5 * track[i] + 0.5 * track[i - 1]
    gate = np.clip(env * 3, 0, 1)
    track[:, 0] *= gate            # silence closes the jaw
    return track


def catmull(points, n=14, closed=True):
    """Dense smooth curve through control points (Catmull-Rom)."""
    pts = list(points)
    if closed:
        pts = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        pts = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = (np.array(pts[j]) for j in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(tuple(
                0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)))
    return out


def lip_y(x, xl, xr, yl, yr, ymid):
    """y on a corner-to-corner lip arc at x (quadratic through 3 points)."""
    t = (x - xl) / max(1e-6, (xr - xl))
    return (1 - t) ** 2 * yl + 2 * (1 - t) * t * ymid + t ** 2 * yr


def face_shapes(draw, W, H, mouth, eyes, wob, color):
    """mouth=(open,width,smile,pucker); eyes=openness 0..1; wob=wobble time."""
    mo, mw_f, smile, pucker = mouth
    s = min(W, H) / 720.0
    j = lambda a, ph: math.sin(wob * a + ph)          # cheap organic noise
    cx = W // 2 + int(4 * s * j(1.3, 0))
    cy = int(H * 0.44 + 5 * s * j(0.9, 2))

    # ---- eyes: slanted curved "mischievous" carve, squash with `eyes`
    ew, eh = 175 * s, 130 * s * max(0.12, eyes)
    for side in (-1, 1):
        ex = cx + side * int(185 * s + 3 * s * j(1.7, side))
        ey = cy - int(160 * s)
        inner = (ex - side * ew * 0.55, ey + eh * 0.45)
        peak = (ex + side * ew * 0.10, ey - eh * 0.55 - 6 * s * j(1.1, side * 3))
        outer = (ex + side * ew * 0.55, ey + eh * 0.10)
        obot = (ex + side * ew * 0.25, ey + eh * 0.50)
        draw.polygon(catmull([inner, peak, outer, obot]), fill=color)
        # angry notch: bite from the inner-bottom edge
        draw.polygon([(ex - side * ew * 0.55, ey + eh * 0.5 + 2),
                      (ex - side * ew * 0.15, ey + eh * 0.05),
                      (ex + side * ew * 0.05, ey + eh * 0.55)], fill=(0, 0, 0))

    # ---- mouth: curvy lips with corner lift, pucker narrows it into an oval
    mw = 500 * s * mw_f * (1 - 0.25 * pucker)
    mh = (36 + mo * 300) * s
    my = cy + int(95 * s)
    lift = smile * 55 * s - pucker * 10 * s
    xl, xr = cx - mw / 2, cx + mw / 2
    yl = my - lift + 4 * s * j(1.5, 1)
    yr = my - lift - 4 * s * j(1.5, 4)
    y_up = my - mh * 0.18 - 8 * s * (1 - pucker)      # upper lip mid
    y_dn = my + mh                                     # lower lip mid
    upper = [(xl, yl), (cx - mw * 0.22, y_up), (cx + mw * 0.22, y_up), (xr, yr)]
    lower = [(xr, yr), (cx + mw * 0.28, y_dn), (cx - mw * 0.28, y_dn), (xl, yl)]
    draw.polygon(catmull(upper + lower[1:-1]), fill=color)

    # ---- teeth: notches hanging from lip curves (skip when puckered)
    if pucker < 0.5:
        n = 4
        th_top = (10 + mo * 30) * s
        th_bot = (8 + mo * 24) * s
        for i in range(n):
            fx = xl + mw * (0.16 + 0.7 * i / (n - 1)) - 26 * s
            y0 = lip_y(fx + 26 * s, xl, xr, yl, yr, y_up)
            draw.rectangle([fx, y0 - 2, fx + 52 * s, y0 + th_top], fill=(0, 0, 0))
        for i in range(n - 1):
            fx = xl + mw * (0.25 + 0.62 * i / max(1, n - 2)) - 26 * s
            y1 = lip_y(fx + 26 * s, xl, xr, yl, yr, y_dn)
            draw.rectangle([fx, y1 - th_bot, fx + 52 * s, y1 + 2], fill=(0, 0, 0))


def render(track, env, wav_path, out_path, style, W, H):
    from PIL import Image, ImageDraw, ImageFilter

    c = STYLES[style]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rng = np.random.RandomState(7)
    blink_at = set()
    t = 0
    while t < len(track):
        t += rng.randint(2 * FPS, 5 * FPS)
        blink_at.update(range(t, t + 4))

    # eye openness: widen on loud moments, occasional blink, slow drift
    e_smooth = np.convolve(env, np.ones(7) / 7, mode="same")

    for i, mouth in enumerate(track):
        wob = i / FPS * 2 * math.pi * 0.7
        eyes = 0.06 if i in blink_at else 0.75 + 0.45 * e_smooth[i] \
            + 0.08 * math.sin(wob * 0.43)
        args = (tuple(mouth), eyes, wob)

        glow = Image.new("RGB", (W, H), (0, 0, 0))
        face_shapes(ImageDraw.Draw(glow), W, H, *args, c["glow"])
        frame = glow.filter(ImageFilter.GaussianBlur(int(min(W, H) * 0.04)))
        halo = Image.new("RGB", (W, H), (0, 0, 0))
        face_shapes(ImageDraw.Draw(halo), W, H, *args, c["mid"])
        frame = Image.blend(frame, halo.filter(ImageFilter.GaussianBlur(int(min(W, H) * 0.012))), 0.6)
        core = Image.new("RGB", (W, H), (0, 0, 0))
        face_shapes(ImageDraw.Draw(core), W, H, *args, c["core"])
        frame = Image.composite(core, frame, core.convert("L").point(lambda v: 255 if v > 10 else 0))
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        sys.exit("ffmpeg failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("--out", default="spooky.mp4")
    p.add_argument("--style", default="pumpkin", choices=list(STYLES))
    p.add_argument("--voice", default="am_michael")
    p.add_argument("--audio", help="use this wav/mp3 instead of TTS (e.g. song vocals)")
    p.add_argument("--size", default="1280x720", help="WxH, e.g. 1920x1080")
    args = p.parse_args()
    W, H = (int(v) for v in args.size.split("x"))

    tokens = None
    if args.audio:
        wav = args.audio
    elif args.text:
        wav = tempfile.mktemp(suffix=".wav")
        print("[1/2] TTS ...")
        _, tokens = tts(args.text, args.voice, wav)
    else:
        sys.exit("give text to speak, or --audio file")

    audio, sr = load_audio(wav)
    n_frames = max(1, int(len(audio) / sr * FPS))
    env = envelope(audio, sr, n_frames)
    if tokens:
        track = mouth_track(tokens, env, n_frames)
    else:
        track = np.stack([env, np.full(n_frames, 0.9),
                          np.full(n_frames, 0.4), np.zeros(n_frames)], axis=1)
    print("[2/2] rendering ...")
    render(track, env, wav, os.path.abspath(args.out), args.style, W, H)
    if not args.audio:
        os.unlink(wav)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
