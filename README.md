# maritime-vision-suite

Computer vision pipeline for sailing race footage, built around a camera
mounted on the mast: sea/sky segmentation, boat detection, and — the
differentiating goal — locating gust/lull patches on the water surface from
their texture, as a tactical aid for routing decisions during a race.

Today the code is a minimal OpenCV frame-reading harness with live FPS
measurement; it's the frame-ingestion layer the stages below will be built
on top of.

## Current state

What's actually implemented right now (`detection/python/src/`):

- `video_loader.py` / `main.py` — opens a video file with OpenCV, reads it
  frame by frame, overlays a live FPS counter, exits on `q`.
- `config.py` — paths and display settings.

**Not implemented yet:** everything described in the roadmap below. This
README describes the target scope; the code above is only the starting
point.

## Roadmap

- [x] **Sea/sky horizon detection (classical baseline)** — see below
- [x] **Sea/sky segmentation (learned model)** — see below; replaces the
  single-line baseline with a CNN that handles coastline/obstacles directly
- [ ] **Boat detection** — YOLO fine-tuned on sailing footage, combining
  [Singapore Maritime Dataset (SMD)](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset)
  and [SeaShips](https://github.com/jiaming-wang/SeaShips) with footage from
  public race broadcasts (America's Cup / SailGP) for the sailing-specific
  gap neither dataset covers
- [ ] **Wind-zone detection** — no dataset or prior work exists for this, so
  it's framed as an **unsupervised** problem, not classification:
  - Segment relatively rougher/darker water patches (gusts/"cat's paws")
    from relatively smoother ones in the *same frame* — local texture
    contrast (Gabor response, high-frequency FFT energy) rather than an
    absolute brightness threshold, which breaks under cloud shadow/glare
  - Add **optical flow** as a second, independent signal: ripple motion
    differs between gust and calm patches even when lighting is deceptive
  - **Validation methodology** (this is the part that actually needs to
    hold up): correlate a flagged patch against the boat's own anemometer
    reading once the boat sails into it, logged over many crossings. No
    external ground truth exists, so this self-supervised comparison is
    the only honest way to report an accuracy number instead of "looks
    right"
  - Known failure modes to document, not hide: sun glare, cloud-shadow
    confounds, mast sway/vibration destabilizing the frame
- [ ] Quantitative wind speed/direction estimation from wave state —
  exploratory, reported with explicit error bars against real wind data,
  not sold as anemometer-equivalent
- [ ] Simple rule-based manoeuvre suggestions (e.g. route toward a detected
  gust, flag a tack on an apparent wind shift) on top of the stages above
- [ ] Latency benchmarks (p50/p99) on target hardware
- [x] Unit tests + CI
- [x] Remove hardcoded local video path, read from `config.py` / CLI arg

## Sea/sky horizon detection (classical baseline)

`segmentation/src/horizon.py` estimates the horizon as a single straight
line per frame: Canny edges, probabilistic Hough transform to find
near-horizontal segments, weighted least-squares fit across them. No
training, no GPU — a measurable starting point before a learned model.

**Limitation by construction:** a straight line can't represent a
coastline or an obstacle crossing the horizon. `evaluate_horizon.py` scores
the detector only on mask columns that are a clean sky-then-water
transition, and reports what fraction of columns actually qualify — so the
number isn't silently inflated by skipping the hard cases.

Measured on the **full** MaSTr1325 dataset, 1325 images
(`scripts/evaluate_horizon.py`):

```
Images evaluated: 414 (skipped: 911 — no clean sky/water columns, e.g. coastline or a large obstacle fills the frame)
Mean fraction of columns with clean ground truth: 31.0%
Pixel error over all valid columns, all images:
  MAE:    16.5 px
  Median: 7.1 px
  Std:    16.8 px
Per-image MAE: mean 15.6 px, worst 77.1 px, best 0.2 px
```

Two things worth being upfront about. First, **69% of the dataset (911/1325
images) has no column this metric can even score** — the straight-line
assumption isn't just imprecise on those, it's inapplicable, and that
fraction is the more honest signal than the pixel error itself. Second,
per-image mean MAE (15.6px) is in the same range as published deep
segmentation baselines on the related MODD2 benchmark (SegNet 13.5px,
PSPNet 13.8px — see [MODD2 leaderboard](https://box.vicos.si/borja/viamaro/index.html))
— but again only on the third of images with an easy, mostly-open-water
scene, and with a long tail: worst case 77.1px. This is a baseline to
beat, not a result to lean on.

(An earlier 50-image dev sample gave a rosier-looking 12.3px mean MAE —
the gap shows why evaluating on a tiny sample before scaling up would have
been misleading.)

Run it yourself:

```bash
python scripts/fetch_mastr1325_sample.py --n 1325   # the whole dataset
python scripts/evaluate_horizon.py
```

**Visual results** — each image is two panels. Left: red is the predicted
horizon, green dots are ground truth but **only drawn on the clean columns
the metric actually scores** (no coastline/obstacles in that column) — on
a coastline-heavy image this can look nearly empty, which is correct, not
missing data. Right: the full ground-truth mask with nothing filtered out
(orange=sky, blue=water, red=obstacle, gray=ignore), so you can always see
what the left panel's green dots are a subset of.

| Best (0.2px) | Median (6.4px) | Worst (77.1px) |
|---|---|---|
| ![best](docs/examples/horizon_best.jpg) | ![median](docs/examples/horizon_median.jpg) | ![worst](docs/examples/horizon_worst.jpg) |

The median example shows a ship sitting on the horizon with a mild slope
mismatch between predicted and ground truth — the kind of error a learned
model should close. The worst case is a large obstacle (a pier/mooring
structure) filling most of the frame under heavy haze: only a thin strip
of open water is left to score, and the line locks onto the haze band well
above the real waterline — visible directly by comparing the red line
against the blue water region on the right panel.

Generate overlays for the whole sample with:

```bash
python scripts/visualize_horizon.py
# writes to data/outputs/horizon_overlays/ (gitignored, regenerate locally)
```

## Sea/sky segmentation (learned model)

`segmentation/src/model.py` is a small U-Net (3 down/up stages, base
width 16, ~1.9M params) trained from scratch — no pretrained weights, no
GPU, on CPU in 22 minutes on the full MaSTr1325 dataset. It predicts a
per-pixel class (obstacle/water/sky) directly, so unlike the single-line
baseline it isn't structurally blind to coastline or obstacles.

Training: 1325 images, 1127 train / 198 val (fixed 85/15 split, seed 42),
30 epochs, Adam, images resized to 192×256, best checkpoint kept by val
mIoU (not just the last epoch — training was noisy, with two epochs
dropping to 0.37 and 0.90 mIoU before recovering, likely the LR being a
bit high for this model size).

```bash
pip install -r requirements-segmentation.txt --index-url https://download.pytorch.org/whl/cpu
python scripts/train_segmentation.py --epochs 30
```

**Results on the full 198-image validation set — 100% of it, not 31%:**

```
Best val mIoU: 0.9810
Per-class IoU:
  obstacle  : 0.9543
  water     : 0.9921
  sky       : 0.9967
Pixel accuracy: 0.9957
```

This is the direct payoff of switching from a geometric assumption to a
learned model: the classical baseline could only be scored on columns
with a clean sky/water transition (31% of the dataset) and still landed
around 15.6px MAE; the CNN is scored on every validation image, full
frame, and reaches 0.981 mIoU. The two numbers aren't on the same scale
(pixel error vs IoU), so treat this as "handles the other 69% of cases
the baseline couldn't touch at all," not a like-for-like percentage
comparison.

**Visual comparison** (predicted vs ground truth, same color coding as
above):

| Best (mIoU 1.000) | Median (mIoU 0.988) | Worst (mIoU 0.787) |
|---|---|---|
| ![best](docs/examples/segmentation_best.jpg) | ![median](docs/examples/segmentation_median.jpg) | ![worst](docs/examples/segmentation_worst.jpg) |

Median shows a complex coastline (trees, buildings, a small buoy) matched
almost pixel-for-pixel — the case that broke the straight-line baseline
outright. Even the worst case is a minor disagreement on exactly where an
obstacle's far edge ends under an ambiguous (ignore-labelled) boundary
region, not a structural failure like the baseline's haze/horizon
confusion.

Run it yourself (needs a trained checkpoint, see above):

```bash
python scripts/visualize_segmentation.py
```

**Known limitations, not hidden:** trained on 1325 images from one
geographic area (gulf of Koper, Slovenia) with a fixed USV camera height
— generalization to a mast-mounted camera on a sailing boat, different
water/lighting conditions, and open-ocean scenes with no coastline at all
is untested, not assumed. The training instability (two mid-training mIoU
drops) also suggests the learning rate schedule is worth revisiting before
this becomes anything more than a baseline for the next stage.

## Datasets

- **[MaSTr1325](https://box.vicos.si/borja/viamaro/index.html)** (CC BY 4.0,
  Bovcon et al., IROS 2019) — 1325 USV-camera images with per-pixel
  sea/sky/obstacle masks. A reproducible 50-image sample is fetched with:

  ```bash
  python scripts/fetch_mastr1325_sample.py
  ```

  This downloads the full dataset into `data/_cache/` (gitignored) and
  copies a fixed-seed sample into `data/datasets/mastr1325_sample/`. 50
  is a quick-start default, not a training set — the full dataset is only
  ~24MB, so for anything beyond a quick look at the baseline, pull more:

  ```bash
  python scripts/fetch_mastr1325_sample.py --n 1325   # the whole dataset
  ```

- **[LaRS](https://lojzezust.github.io/lars-dataset/)** (ICCV 2023) — larger
  and more diverse panoptic maritime dataset, gated behind a request form
  (name/email/affiliation/intended use). Not yet requested.

Neither dataset has sailing-specific gust/wind-zone annotations — no such
dataset exists (see roadmap above); that stage is unsupervised by design.

## Requirements

```
numpy==2.2.6
opencv-python==4.12.0.88
scipy==1.15.3
```

## Running

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python detection/python/src/main.py
```

Place an input video at `data/videos/sample.mp4` (path configurable in
`config.py`).

## Testing

```bash
pip install pytest ruff
ruff check detection/python
pytest detection/python -v
```

Runs on every push/PR via GitHub Actions (`.github/workflows/ci.yml`),
on Python 3.10 and 3.11.

## License

MIT — see [LICENSE](LICENSE).
