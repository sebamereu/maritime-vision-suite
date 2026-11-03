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

- [ ] **Sea/sky segmentation** — horizon detection, using
  [MaSTr1325](https://box.vicos.si/borja/viamaro/index.html) / [LaRS](https://lojzezust.github.io/lars-dataset/)
  (semantic segmentation of sea/sky/obstacles from USV-mounted cameras —
  close match for a mast-camera viewpoint)
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

## Datasets

- **[MaSTr1325](https://box.vicos.si/borja/viamaro/index.html)** (CC BY 4.0,
  Bovcon et al., IROS 2019) — 1325 USV-camera images with per-pixel
  sea/sky/obstacle masks. A reproducible 50-image sample is fetched with:

  ```bash
  python scripts/fetch_mastr1325_sample.py
  ```

  This downloads the full dataset into `data/_cache/` (gitignored) and
  copies a fixed-seed sample into `data/datasets/mastr1325_sample/`.

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
