# Faces Movie Maker

Turn text or songs into talking-face videos — glowing Halloween projection
faces, cartoon characters, or your own drawn faces — fully local, no API keys.

Built for projecting onto pumpkins, windows, and inflatables (think
"singing jack-o'-lantern"), but the sprite engine animates any face you can
draw as a 3×3 phoneme sheet.

## What's here

| Tool | What it does |
|---|---|
| `webapp/` | **The studio** — browser UI: store phoneme sheets, render from text (TTS) or from an audio file + timed lyrics |
| `sprite.py` | Sprite-sheet engine: phoneme-timed mouth poses, smooth SDF morphing, squash & stretch, random blinks |
| `pumpkin.py` | Procedurally drawn jack-o'-lantern / ghost face (no sheet needed) |
| `ghost.py` | Animate a single cartoon image by stretching its mouth region |
| `talk.py` | Photo-realistic talking head from one photo (requires SadTalker, see below) |

## Setup

Requires Python 3.10+, [ffmpeg](https://ffmpeg.org), and (for the default
English voices) espeak-ng.

```bash
# macOS
brew install ffmpeg espeak-ng

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

First run downloads the Kokoro TTS model (~300 MB) from HuggingFace.

## The studio (web app)

```bash
.venv/bin/python webapp/app.py
# open http://localhost:5173
```

1. **Upload a phoneme sheet** — a 3×3 grid image of the same face with nine
   mouth positions, in this order:

   ```
   AI   E    FV
   L    MBP  O
   U    WQ   etc/rest
   ```

   Black-on-white ink or full-color artwork both work. (Image generators
   produce these well — ask for "the exact same face 9 times in a 3x3 grid,
   only the mouth changes, labeled AI/E/FV/L/MBP/O/U/WQ/rest".)

2. **Speak mode** — type text, pick a voice, render.

3. **Sing mode** — upload audio (mp3/wav/m4a) plus timing:
   - **xLights `.xtiming` file** (best): a phoneme layer (AI/E/FV/L/MBP/O/U/WQ/etc
     labels) drives the mouth frame-accurately; a words layer is phonemized
     inside each word's window.
   - **or timed lyric lines**; each line's words are spread across its window
     and the audio's loudness gates the mouth:

     ```
     [00:01.2] I want it that way
     [00:05.8] Tell me why
     ```

**Render styles**: `pumpkin` (glowing orange carve), `ghost` (white glow),
`color` (keeps your sheet's original artwork).

## CLI

```bash
.venv/bin/python sprite.py "This is a test" --sheet myface.png --style color --out test.mp4
.venv/bin/python pumpkin.py "Happy Halloween!" --style ghost --out boo.mp4
```

## Optional: photo-realistic mode (`talk.py`)

`talk.py` animates a real photo via [SadTalker](https://github.com/OpenTalker/SadTalker)
(not included). Clone it into this directory, download its checkpoints, and
install its requirements into the same venv. On Apple Silicon you'll need
small patches for MPS (legacy `tensor.type()` strings); on NVIDIA it works
as-is.

## Notes

- Everything renders locally; the web app binds to 127.0.0.1 only.
- Only use faces, voices, and music you have rights to.
