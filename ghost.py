#!/usr/bin/env python3
"""Text -> talking MP4 from a CARTOON face image (no ML face model).

For faces SadTalker can't detect (ghosts, jack-o'-lanterns, abstract cartoons).
Animates the image directly: the mouth region stretches open in sync with the
TTS audio's loudness envelope, plus a gentle idle bob. 720x1280 @ 30fps.

Usage:
    .venv/bin/python ghost.py "Happy Halloween" --image face.png --out ghost.mp4 \
        --mouth 18,52,133,100    # mouth box x1,y1,x2,y2 in source-image pixels
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


def tts(text, voice, wav_path):
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=voice[0])
    chunks = [audio for (_, _, audio) in pipeline(text, voice=voice)]
    audio = np.concatenate(chunks)
    sf.write(wav_path, audio, 24000)
    return audio, 24000


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


def render(image_path, mouth, env, wav_path, out_path, max_drop=130):
    from PIL import Image

    src = Image.open(image_path).convert("RGB")
    scale = W / src.width
    base = src.resize((W, int(src.height * scale)), Image.LANCZOS)
    bg = src.getpixel((2, 2))  # match canvas to the image's corner color

    mx1, my1, mx2, my2 = [int(v * scale) for v in mouth]
    y_off = (H - base.height) // 2

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i, amp in enumerate(env):
        frame = Image.new("RGB", (W, H), bg)
        face = base.copy()
        drop = int(amp * max_drop)
        if drop > 0:
            strip = face.crop((mx1, my1, mx2, my2))
            stretched = strip.resize((mx2 - mx1, (my2 - my1) + drop), Image.LANCZOS)
            face.paste(stretched, (mx1, my1))
        bob = int(5 * math.sin(i / FPS * 2.0))
        frame.paste(face, (0, y_off + bob))
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        sys.exit("ffmpeg failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="ghost.mp4")
    p.add_argument("--voice", default="am_michael")
    p.add_argument("--mouth", default="18,52,133,100",
                   help="mouth box x1,y1,x2,y2 in source pixels")
    p.add_argument("--max-drop", type=int, default=130,
                   help="max jaw-open in output pixels")
    args = p.parse_args()

    mouth = [int(v) for v in args.mouth.split(",")]
    wav = tempfile.mktemp(suffix=".wav")
    print("[1/2] TTS ...")
    audio, sr = tts(args.text, args.voice, wav)
    n_frames = max(1, int(len(audio) / sr * FPS))
    print("[2/2] rendering ...")
    render(args.image, mouth, envelope(audio, sr, n_frames), wav,
           os.path.abspath(args.out), args.max_drop)
    os.unlink(wav)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
