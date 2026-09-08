# Improving animation quality

Running notes on how the face animation looks and what would make it better.
Kept so we don't re-litigate settled questions or re-try dead ends.

## Where things stand

The renderer is a pure NumPy/PIL compositor — no model draws anything at render
time. It slices a 3x3 sheet into 9 mouth poses and builds frames from them.

| Piece | Current behaviour |
|---|---|
| Pose timing | Phoneme-driven (`sprite_track`), or frame-accurate from an xtiming phoneme layer |
| Transitions, glow styles | Signed-distance-field shape morph (`sdf`) — true intermediate shapes |
| Transitions, color style | Per-pair layered SDF morph (`RegionMorpher`) — the changed region is split into flat-colour layers by k-means, each layer's *shape* is interpolated, then feathered back into untouched artwork |
| Transition length | `T = 3` in `render_video_color` — two in-between frames, ~100 ms |
| Output rate | `FPS = 30` (`pumpkin.py`) |
| Idle life | Sub-pixel drift + slight rotation every frame; squash/stretch from the audio envelope |
| Blinks | Glow styles only — random every 2–5 s by squashing the eye region of the SDF |

## Settled — do not revisit

- **Cross-dissolving two poses.** Shows both mouths at partial opacity — reads as
  a blurry double mouth. Rejected twice. Any blend must *complete* into a pure
  pose, and ideally interpolate geometry rather than pixels.
- **ffmpeg `minterpolate` (mci/aobmc) between poses.** Tested on a Santa pose
  pair; produced the same ghosting as a dissolve. Optical-flow estimators fail on
  flat cartoon art with no texture to track.
- **Hard cuts with no in-betweens.** Clean, but the mouth teleports.
- **Integer-pixel idle motion.** Held poses became pixel-identical for several
  frames, which reads as low frame rate even though the file is 30 fps.

## Open opportunities

Ordered by perceptual gain per unit of effort. Nothing here is blocked by
hardware — see Resources below.

- [ ] **1. Bigger frame budget.** Raise `FPS` to 60 and lengthen `T` so each
      mouth change gets 5–7 in-betweens instead of 2. Roughly 2x render time,
      no new dependencies. Try this before anything cleverer: more samples of a
      decent morph usually beats fewer samples of a great one.

- [ ] **2. Co-articulation.** *Biggest expected win.* Today the mouth holds a
      discrete pose then transitions to the next discrete pose. Real speech
      anticipates upcoming sounds and, when fast, never fully reaches a target
      before moving toward the next. Model the mouth as a smooth path through
      pose space — weight previous/current/next phoneme, undershoot brief
      targets. Pure maths, no new dependencies, no render cost.

- [ ] **3. Denser sprite sheets.** Nine poses is a small vocabulary, so every
      transition has to invent a lot. Half-open variants (between AI and MBP,
      mid-O) shorten the synthesis distance and improve results from *any*
      interpolation method. Costs image-generation time, not engineering.

- [ ] **4. Neural frame interpolation on the GPU.** Only if 1–3 leave something
      wanted. **FILM** is the right model (built for *large* motion between
      frames, which is exactly a pose change); RIFE is faster but assumes small
      motion. Runs on MPS. Adds a dependency and render time — skipping it keeps
      renders fast, so treat it as a last resort rather than the goal.

### Blinks (separate from mouth quality)

- [ ] **Timed blinks from the xtiming file.** Treat an EffectLayer whose labels
      are mostly `blink`/`wink`/`eyes closed` as a blink track, so blinks can be
      authored in xLights alongside the light show and land on the beat.
- [ ] **Blink rate control** in the studio UI (average seconds between blinks, or
      off). Human cadence is ~3–5 s; spooky characters read better at 6–8 s.
- [ ] **Blinks for the color style.** Best: accept a closed-eyes pose (10th cell
      or a second upload) so blinks cut to real artwork. Fallback with no extra
      art: detect the pupil blobs and vertically squash that region, the same
      trick the glow styles use.

## Resources available

The renderer currently uses none of the machine's parallel capacity.

- **GPU:** Apple M4 Max, idle during renders. PyTorch 2.13 with MPS is installed
  and working (it was set up for SadTalker).
- **CPU/RAM:** 6+ cores unused — frame generation is embarrassingly parallel and
  could be pooled; 64 GB RAM.
- **Libraries already present:** OpenCV 4.11 (DIS optical flow, if we ever want
  warp-based morphing), SciPy, Pillow, ffmpeg.
- **Aligner:** AutoLyrixAlign on node7:3001 for word timings. Note it peaks near
  14.5 GB, so don't schedule heavy work on that box beside it.

## How to judge a change

Render the same line before and after and compare frames, rather than trusting a
description of the algorithm:

```bash
ffmpeg -y -i clip.mp4 -vf "fps=8,scale=320:-1,tile=6x3" -frames:v 1 grid.png
```

Watch for: two mouths visible at once (blending too long), identical consecutive
frames (motion stalled), and whether the mouth *travels* between poses or fades.
