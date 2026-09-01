#!/usr/bin/env python3
"""Text -> talking-face MP4, fully local.

Usage:
    .venv/bin/python talk.py "Hello, this is a test." --image face.jpg --out hello.mp4
    .venv/bin/python talk.py "..." --image face.jpg --out x.mp4 --voice am_adam --enhance

Pipeline: Kokoro TTS (text -> wav) -> SadTalker (photo + wav -> mp4).
First run downloads the Kokoro model (~300 MB) from HuggingFace.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SADTALKER = os.path.join(HERE, "SadTalker")
PYTHON = sys.executable

# Voices: af_heart / af_bella (female US), am_adam / am_michael (male US),
# bf_emma (female UK), bm_george (male UK). Full list in Kokoro docs.


def tts(text: str, voice: str, wav_path: str) -> None:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=voice[0])  # 'a' US / 'b' UK English
    chunks = [audio for (_, _, audio) in pipeline(text, voice=voice)]
    sf.write(wav_path, np.concatenate(chunks), 24000)


def sadtalker(image: str, wav_path: str, out_path: str, args) -> None:
    workdir = tempfile.mkdtemp(prefix="sadtalker_")
    cmd = [
        PYTHON, "inference.py",
        "--driven_audio", wav_path,
        "--source_image", os.path.abspath(image),
        "--result_dir", workdir,
        "--preprocess", args.preprocess,
        "--size", str(args.size),
    ]
    if args.still:
        cmd.append("--still")
    if args.enhance:
        cmd += ["--enhancer", "gfpgan"]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    subprocess.run(cmd, cwd=SADTALKER, env=env, check=True)
    results = sorted(glob.glob(os.path.join(workdir, "**", "*.mp4"), recursive=True),
                     key=os.path.getmtime)
    if not results:
        sys.exit("SadTalker produced no mp4 (see log above)")
    shutil.move(results[-1], out_path)
    shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", help="what the face should say")
    p.add_argument("--image", required=True, help="source photo of the face")
    p.add_argument("--out", default="out.mp4", help="output mp4 path")
    p.add_argument("--voice", default="af_heart", help="Kokoro voice id")
    p.add_argument("--preprocess", default="crop", choices=["crop", "resize", "full"],
                   help="crop = isolated face (default)")
    p.add_argument("--size", type=int, default=256, choices=[256, 512])
    p.add_argument("--still", action="store_true", default=True,
                   help="less head motion (default on)")
    p.add_argument("--no-still", dest="still", action="store_false")
    p.add_argument("--enhance", action="store_true",
                   help="GFPGAN face enhancement (sharper, ~2x slower)")
    args = p.parse_args()

    wav = tempfile.mktemp(suffix=".wav")
    print(f"[1/2] TTS ({args.voice}) ...")
    tts(args.text, args.voice, wav)
    print(f"[2/2] SadTalker ({args.preprocess}, {args.size}px) ...")
    sadtalker(args.image, wav, os.path.abspath(args.out), args)
    os.unlink(wav)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
