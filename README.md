# maritime-vision-suite

Work-in-progress pipeline for maritime video analysis: today it's a minimal
OpenCV frame-reading harness with live FPS measurement, laid out as the
scaffold for the detection and event-based tracking stages described below.

## Current state

What's actually implemented right now (`detection/python/src/`):

- `video_loader.py` / `main.py` — opens a video file with OpenCV, reads it
  frame by frame, overlays a live FPS counter, exits on `q`.
- `config.py` — paths and display settings.

**Not implemented yet:** object detection and event-based tracking. The repo
name and this README describe the target scope; the code below is the
frame-ingestion layer everything else will be built on.

## Roadmap

- [ ] Object detection stage (YOLO/ONNX Runtime) on video frames
- [ ] Event-based camera ingestion and tracking
- [ ] Latency benchmarks (p50/p99) on target hardware
- [ ] Unit tests + CI
- [ ] Remove hardcoded local video path, read from `config.py` / CLI arg

## Requirements

```
numpy==2.2.6
opencv-python==4.12.0.88
scipy==1.16.3
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

## License

MIT — see [LICENSE](LICENSE).
