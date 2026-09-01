#!/usr/bin/env python3
"""One-off: sustained AY - EEE - EYE - OH - YOU vowel demo for mouth-shape review."""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
from kokoro import KPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pumpkin import FPS, envelope, mouth_track, render

SR = 24000
GAP = 0.5  # silence between vowels, seconds

VOWELS = [
    ("A", "[A](/ˈAAA/)"),     # long A (eɪ)
    ("E", "[E](/ˈiiii/)"),    # eeee
    ("I", "[I](/ˈIII/)"),     # eye (aɪ)
    ("O", "[O](/ˈOOO/)"),     # oh (oʊ)
    ("U", "[U](/jˈuuu/)"),    # you
]

pipeline = KPipeline(lang_code="a")
audio_parts, tokens, offset = [], [], 0.0
for name, markup in VOWELS:
    for r in pipeline(markup, voice="am_michael"):
        audio_parts.append(r.audio)
        for t in r.tokens:
            if t.start_ts is not None and t.phonemes:
                tokens.append((offset + t.start_ts, offset + t.end_ts, t.phonemes))
                print(f"{name}: {t.phonemes}  {offset + t.start_ts:.2f}-{offset + t.end_ts:.2f}s")
        offset += len(r.audio) / SR
    audio_parts.append(np.zeros(int(GAP * SR)))
    offset += GAP

audio = np.concatenate(audio_parts)
wav = tempfile.mktemp(suffix=".wav")
sf.write(wav, audio, SR)

n_frames = int(len(audio) / SR * FPS)
env = envelope(audio, SR, n_frames)
track = mouth_track(tokens, env, n_frames)
out = "aeiou.mp4"
render(track, env, wav, out, "pumpkin", 1280, 720)
os.unlink(wav)
print("done:", out)
