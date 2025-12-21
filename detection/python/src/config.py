# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "sample.mp4"

WINDOW_NAME = "Maritime Vision Suite"
EXIT_KEY = ord('q')
SHOW_FPS = True
