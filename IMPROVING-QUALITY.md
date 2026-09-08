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

- **Asking for more artwork.** The input stays at the 9 Papagayo visemes
  (AI/E/FV/L/MBP/O/U/WQ/rest). Denser sheets, extra half-open poses and separate
  closed-eye artwork are off the table — not for xLights compatibility (the
  deliverable is a video; we are not driving the xLights Faces module), simply
  because the user should hand over nine images and nothing more. Everything
  else is ours to synthesise.

## Open opportunities

Ordered by perceptual gain per unit of effort. Nothing here is blocked by
hardware — see Resources below. All of it works within the fixed 9-pose sheet.

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

- [ ] **3. Better morph maths — not more materialised poses.** Producing the
      in-betweens ourselves is already the architecture (`RegionMorpher`), and
      it is the only route open to us. But note what it can and cannot buy:

      *Baking computed midpoints into the library as extra poses is a caching
      optimisation, not a quality one.* A geometric midpoint of AI and MBP holds
      no information the two endpoints didn't already have — animating
      AI → midpoint → MBP renders the identical pixels as morphing AI → MBP
      across the same frames. A hand-drawn half-open mouth would have added
      information (the artist knows what one looks like); an interpolated one
      does not.

      So quality has to come from a better *path* between poses, not more stops
      along it: correct the morph so shapes travel plausibly (a mouth closes
      from the edges inward, the lower lip moves further than the upper), rather
      than every colour layer shrinking uniformly toward its target.

- [ ] **4. Neural frame interpolation on the GPU.** Only if 1–3 leave something
      wanted. **FILM** is the right model (built for *large* motion between
      frames, which is exactly a pose change); RIFE is faster but assumes small
      motion. Runs on MPS. Adds a dependency and render time — skipping it keeps
      renders fast, so treat it as a last resort rather than the goal.

      This is the one option that genuinely *adds* information rather than
      redistributing what the nine poses already contain, because the model has
      learned what moving mouths look like. A generative variant (an image model
      inventing true intermediate artwork once per sheet, cached) would add even
      more — but style consistency across generated cells is the hard part, and
      an off-model frame is far more jarring than a slightly stiff morph. Park
      both until the cheap wins are exhausted.

### Blinks (separate from mouth quality)

- [ ] **Timed blinks from the xtiming file.** Treat an EffectLayer whose labels
      are mostly `blink`/`wink`/`eyes closed` as a blink track, so blinks can be
      authored in xLights alongside the light show and land on the beat.
- [ ] **Blink rate control** in the studio UI (average seconds between blinks, or
      off). Human cadence is ~3–5 s; spooky characters read better at 6–8 s.
- [ ] **Blinks for the color style** (currently glow-only). Since we don't ask
      for closed-eye artwork, do it synthetically: find the pupil blobs in the
      upper face and vertically squash that region of the RGB art, the same
      row-remap the glow styles already use on the distance field. The eyes are
      identical across all nine cells, so the region only has to be found once
      per sheet.

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
