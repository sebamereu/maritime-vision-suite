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
- [x] **Boat detection (YOLOv8n, zero-shot baseline + fine-tuned on
  SeaShips)** — see below. [Singapore Maritime Dataset
  (SMD)](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset)
  and footage from public race broadcasts (America's Cup / SailGP) are not
  yet integrated — see Known limitations below
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

## Boat detection

`boat_detection/src/model.py` wraps [ultralytics
YOLOv8n](https://github.com/ultralytics/ultralytics) — COCO-pretrained,
~3M params, already ships a "boat" class. `boat_detection/src/dataset.py`
converts [SeaShips(7000)](#datasets)'s Pascal VOC annotations to YOLO
labels, collapsing its six ship-type classes (cargo carrier, container
ship, fishing boat, etc.) into one: this project asks "is there a boat",
not ship taxonomy.

Measured the same way as the horizon baseline — a number before any
training, not a result to lean on — on a fixed-seed 1500-image sample
(1200 train / 300 val, 396 ground-truth boxes), using ultralytics' own
`val()` rather than a hand-rolled metric (see below for why):

```
                              mAP50    mAP50-95   Precision   Recall
Zero-shot (stock yolov8n.pt)  0.945     0.616      1.000      0.944
Fine-tuned (40 epochs)        0.995     0.774      1.000      0.995
```

```bash
pip install -r requirements-detection.txt --index-url https://download.pytorch.org/whl/cpu
python scripts/fetch_seaships_sample.py --n 1500
python scripts/evaluate_boat_detection.py                                            # zero-shot
python scripts/train_boat_detection.py --epochs 40 --checkpoint data/checkpoints/yolov8n_boats.pt
python scripts/evaluate_boat_detection.py --checkpoint data/checkpoints/yolov8n_boats.pt
```

**mAP alone understates what fine-tuning actually fixed.** An earlier run
on a much smaller sample (120 train / 30 val — still in the git history)
showed fine-tuned and zero-shot landing on nearly identical mAP50
(0.975 both), which on its own reads as "fine-tuning didn't help." It
missed the real story, visible only by checking predictions at a fixed,
practical confidence threshold (0.25 — what you'd actually use, not the
threshold-sweep mAP integrates over) instead of trusting the aggregate
metric alone:

```
                          images with zero       images with at least
                          detections @conf=0.25  one missed box (IoU<0.5)
Zero-shot                 34%                     61%
Fine-tuned (120 train)    40%                     ~53%
Fine-tuned (1200 train)    0%                      3%
```

The zero-shot model *can* localize most SeaShips boats (that's what the
high mAP measures, since mAP sweeps every confidence level including
near-zero ones) but is rarely *confident* about them — COCO's "boat"
class never saw this specific port/hull style, so at a threshold you'd
actually deploy with, it misses most of the time. Fine-tuning on only
120 images didn't fix that calibration problem (not enough signal to
learn confident predictions, same root cause as the still-close mAP).
Fine-tuning on 1200 images did: zero images with no detection at all,
and the model's confidence is now trustworthy at a normal operating
threshold, not just somewhere on the full sweep. This is the more
honest headline number for "does fine-tuning work here" than mAP alone
— confirmed by rerunning the exact same check, nothing cherry-picked.

**On the metric itself:** an earlier version of this evaluation used a
hand-rolled IoU-matching AP (same approach as `evaluate_horizon.py`). It
turned out to be mathematically correct — verified line-for-line against
ultralytics' own `ap_per_class` fed identical inputs — but wildly
unstable on a validation set this small (34 boxes): a handful of
overlapping candidate boxes on a couple of cluttered images (one
elongated ore carrier spanning almost the full frame produced ~140
overlapping candidate boxes) can flip which single detection gets
credited as *the* match, swinging the reported number by 50+ points
(0.43 vs 0.975 on the exact same checkpoint) without the model changing
at all. Rather than present a self-built metric with that much variance
as a real result, `evaluate_boat_detection.py` now uses ultralytics' own
validator directly — including for the zero-shot case, via a second
label set that relabels ground truth at COCO's own "boat" class index so
the un-fine-tuned 80-class model can be scored fairly. This instability
is itself the more honest takeaway than either individual number: mAP on
a 30-image eval set is noisy, full stop.

**Visual comparison** (red: predicted, green: ground truth, the
1200-image fine-tuned checkpoint):

| Best (IoU 0.99) | Median (IoU 0.88) | Worst (IoU 0.43) |
|---|---|---|
| ![best](docs/examples/boat_detection_best.jpg) | ![median](docs/examples/boat_detection_median.jpg) | ![worst](docs/examples/boat_detection_worst.jpg) |

The worst case here is a real but minor localization error, not a miss:
two ships close together, a container ship and a smaller vessel further
back, where the model only boxes the nearer one — visible by comparing
against the two separate ground-truth boxes on the right. Median is now
0.88 IoU, not 0.36 — a direct, visible consequence of the confidence-
calibration fix above, not just a better aggregate number.

**Known limitations, not hidden:** SeaShips is shore-camera footage of
commercial shipping at a fixed Chinese port — a real domain gap from a
mast-mounted camera on a sailing boat in open water (different hull
shapes, no recreational sailboats at all, fixed camera vs. a moving,
vibrating mast). SMD and actual race-broadcast footage (America's Cup /
SailGP) are not yet integrated: SMD ships as raw multi-GB video with a
custom per-frame ground-truth format (horizon + tracking + detection
combined), which needs its own frame-extraction and parsing pipeline —
a separate chunk of work from this stage. The numbers above use `--n
1500 --epochs 40`, not the scripts' lighter quick-start defaults (`--n
150`, `--epochs 30`) — this page is itself the evidence that the
lighter defaults under-train the model's confidence calibration, so
anyone reproducing a real result here needs the larger flags; both
scale up further still.

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

- **[SeaShips(7000)](https://sites.ucmerced.edu/files/wdu/files/seaship.pdf)**
  (Shao et al., IEEE Transactions on Multimedia, 2018) — 7000 images of
  commercial ships at a Chinese coastal port, Pascal VOC boxes across six
  ship-type classes. The original paper has no clearly published
  redistribution license of its own; this project fetches a [public
  Hugging Face mirror](https://huggingface.co/datasets/datnguyentien204/Seaship7000)
  for the same academic/portfolio research use the dataset is normally
  cited for. A reproducible sample is fetched with (`--n 150` is a
  quick-start default; the boat detection results above use `--n 1500`,
  since the smaller sample under-trains the model's confidence):

  ```bash
  python scripts/fetch_seaships_sample.py --n 1500
  ```

None of these datasets have sailing-specific gust/wind-zone annotations —
no such dataset exists (see roadmap above); that stage is unsupervised by
design.

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
pip install -r requirements-segmentation.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -r requirements-detection.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

ruff check detection/python segmentation scripts boat_detection

# Run these separately, not combined into one `pytest a b c` call:
# segmentation/src/ and boat_detection/src/ both have same-named modules
# (model.py, dataset.py, metrics.py) added to sys.path by their
# conftest.py; collected together, the second package's imports
# silently resolve to the first package's modules.
pytest detection/python -v
pytest segmentation/tests -v
pytest boat_detection/tests -v
```

Runs on every push/PR via GitHub Actions (`.github/workflows/ci.yml`),
on Python 3.10 and 3.11.

## License

MIT — see [LICENSE](LICENSE).
