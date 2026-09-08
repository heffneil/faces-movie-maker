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
- **Animating every phoneme.** Raw phoneme timings give a new viseme every two
  or three frames — measured on ordinary TTS speech, 35% of segments lasted
  67 ms and 67% were under 130 ms. The mouth never arrives anywhere and the
  result is jittery. A minimum hold (`enforce_min_hold`, ~150 ms) is not a
  compromise, it is what lip-sync artists do: animate at roughly syllable rate,
  absorbing brief phonemes into their neighbours.

- **Anything computed per pose pair.** The palette and region box used to be
  derived per pair, so both shifted at every morph-target change — about twelve
  times a second in speech, which reads as flicker. Region, palette and shape
  decomposition are all per SHEET now. Anything new that varies per pair needs
  the same scrutiny.

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

- [x] **1. Bigger frame budget.** *Done 2026-09-02.* Frame rate is a parameter
      (`--fps`, and an FPS control in both studio panels) rather than a module
      constant; 30 stays the default, 60 costs about 2x render time. Note the
      co-articulation window below is specified in milliseconds, so the
      animation feels identical at either rate — 60 just samples it more finely.

- [x] **2. Co-articulation.** *Done 2026-09-02 — `coarticulate()`.* Replaced the
      hold-then-transition model in both render paths. Each phoneme segment
      holds full influence over its own span and decays either side over a fixed
      ~50 ms window, so neighbours overlap; every frame is then a morph between
      the two strongest targets. Falls out of that for free:

      - *anticipation* — the mouth is already ~30% toward a sound two frames
        before it starts
      - *undershoot* — a 2–3 frame phoneme only travels ~56–70% toward its pose
        because its neighbours dilute it, exactly as a real mouth does in fast
        speech; a long phoneme still settles exactly on its pose
      - *continuity* — at a crossover both orderings agree at w=0.5, so there is
        no discontinuity, and no frame is ever a static held pose

      Verify with the synthetic track in the commit message, or by checking that
      consecutive frames always differ (see "How to judge a change").

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
      along it.

      *Partly done 2026-09-02* — `RegionMorpher.frame(..., asym=True)` no longer
      advances every pixel of the region at the same rate. The morph weight is
      now a field biased so the lower lip travels further than the upper, and so
      the corners lead when the mouth is closing (the centre leads when it is
      opening — direction is detected by whether the dark interior cluster is
      shrinking). The bias is scaled by `4w(1-w)`, which vanishes at w=0 and
      w=1, so both endpoints still land exactly on their pose.

      Still open: the layers move independently, so a tongue or teeth can drift
      slightly out of step with the lip that should carry them. Grouping layers
      that belong to the same anatomical part would fix it.

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

### Numbers worth re-measuring after any change

Measured on one TTS line ("Merry Christmas everybody!..."), 30 fps:

| | before min-hold | after |
|---|---|---|
| morph-target changes | 12.2 / sec | ~4 / sec |
| frames landed on a pose (w<0.05) | 24% | ~51% |
| frames stuck mid-morph (w>0.35) | 47% | ~24% |

The knobs are `min_hold_ms` (150) and `co_ms` (28) in `coarticulate`. Raising
the hold calms the motion further but eventually costs sync fidelity, since
absorbed visemes move boundaries by up to one hold; `/tmp` sweep script in the
commit history shows the trade-off curve.

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
