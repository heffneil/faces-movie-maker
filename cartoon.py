#!/usr/bin/env python3
"""Text -> cartoon ghost-face talking MP4 (amplitude-driven jaw, no ML face model).

Usage:
    .venv/bin/python cartoon.py "Hello there" --out ghost.mp4 [--voice am_adam]

Draws a glowing white cartoon face (arc eyes, toothy grin) and animates the
jaw from the TTS audio's loudness envelope — the style of the viral
"talking ghost/pumpkin face" meme videos. 720x1280 @ 30fps.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

W, H = 720, 1280
FPS = 30


def tts(text: str, voice: str, wav_path: str):
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=voice[0])
    chunks = [audio for (_, _, audio) in pipeline(text, voice=voice)]
    audio = np.concatenate(chunks)
    sf.write(wav_path, audio, 24000)
    return audio, 24000


def envelope(audio, sr, n_frames):
    # per-video-frame RMS loudness, normalized 0..1 with a noise gate
    hop = len(audio) / n_frames
    env = np.array([
        np.sqrt(np.mean(audio[int(i * hop):max(int(i * hop) + 1, int((i + 1) * hop))] ** 2))
        for i in range(n_frames)
    ])
    env = np.convolve(env, np.ones(3) / 3, mode="same")  # smooth
    peak = env.max() or 1.0
    env = env / peak
    env[env < 0.08] = 0.0
    return env


def draw_frame(draw, amp, blink):
    from PIL import ImageDraw  # noqa: F401  (type hints only)

    cx, cy = W // 2, H // 2
    fw, fh = 640, 800  # face blob size

    # glowing white face blob
    draw.rounded_rectangle([cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 2],
                           radius=280, fill=(245, 248, 252))

    line = (168, 178, 192)  # soft gray features, like the reference
    lw = 26

    # eyes: angled arcs (angry brows). Blink -> flat lines.
    for side in (-1, 1):
        ex = cx + side * 150
        ey = cy - 190
        if blink:
            draw.line([ex - 90, ey, ex + 90, ey], fill=line, width=lw)
        else:
            box = [ex - 100, ey - 60, ex + 100, ey + 90]
            start, end = (200, 320) if side < 0 else (220, 340)
            draw.arc(box, start=start, end=end, fill=line, width=lw)

    # mouth: wide grin; jaw (bottom edge) drops with amplitude
    mw = 440
    top = cy + 60
    base_h = 120
    drop = int(amp * 170)
    bot = top + base_h + drop
    draw.rounded_rectangle([cx - mw // 2, top, cx + mw // 2, bot],
                           radius=60, fill=(70, 78, 92), outline=line, width=14)
    # teeth: fixed top row; bottom row rides the jaw
    tooth_w, tooth_h, gap = 70, 46, 14
    n = 5
    total = n * tooth_w + (n - 1) * gap
    x0 = cx - total // 2
    for i in range(n):
        x = x0 + i * (tooth_w + gap)
        draw.rounded_rectangle([x, top + 10, x + tooth_w, top + 10 + tooth_h],
                               radius=12, fill=(245, 248, 252))
        draw.rounded_rectangle([x, bot - 10 - tooth_h, x + tooth_w, bot - 10],
                               radius=12, fill=(245, 248, 252))


def render(env, wav_path, out_path):
    from PIL import Image, ImageDraw, ImageFilter

    ff = "ffmpeg"
    proc = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rng = np.random.RandomState(7)
    blink_at = set()
    t = 0
    while t < len(env):  # a blink every ~2-4s, lasting 4 frames
        t += rng.randint(2 * FPS, 4 * FPS)
        blink_at.update(range(t, t + 4))

    for i, amp in enumerate(env):
        img = Image.new("RGB", (W, H), (16, 20, 30))
        # soft glow behind the face
        glow = Image.new("RGB", (W, H), (16, 20, 30))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([W // 2 - 380, H // 2 - 480, W // 2 + 380, H // 2 + 480],
                   fill=(90, 110, 140))
        img = Image.blend(img, glow.filter(ImageFilter.GaussianBlur(120)), 0.55)
        d = ImageDraw.Draw(img)
        # gentle idle bob
        bob = int(6 * math.sin(i / FPS * 2.2))
        d2 = ImageDraw.Draw(img)
        draw_frame(d2, amp, blink=(i in blink_at))
        img = img.transform((W, H), Image.AFFINE, (1, 0, 0, 0, 1, -bob))
        proc.stdin.write(img.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        sys.exit("ffmpeg failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text")
    p.add_argument("--out", default="cartoon.mp4")
    p.add_argument("--voice", default="am_michael")
    args = p.parse_args()

    wav = tempfile.mktemp(suffix=".wav")
    print("[1/2] TTS ...")
    audio, sr = tts(args.text, args.voice, wav)
    n_frames = max(1, int(len(audio) / sr * FPS))
    print("[2/2] rendering frames ...")
    render(envelope(audio, sr, n_frames), wav, os.path.abspath(args.out))
    os.unlink(wav)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
