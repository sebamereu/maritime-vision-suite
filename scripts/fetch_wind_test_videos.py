#!/usr/bin/env python3
"""Download two free-license aerial water videos from Pixabay for
testing wind_detection/ -- not training data (the detector is
unsupervised, see wind_detection/src/detector.py), just footage that
actually shows the phenomenon (visibly rougher vs. smoother patches of
open water) to run the detector on.

Pixabay License: free for commercial and non-commercial use, no
attribution required.

Usage:
    python scripts/fetch_wind_test_videos.py
"""
import urllib.request
from pathlib import Path

USER_AGENT = "maritime-vision-suite-research/1.0 (portfolio project; contact: sebamere00@gmail.com)"

VIDEOS = [
    ("sea_texture.mp4",
     "https://cdn.pixabay.com/video/2018/01/21/13864-252799058_small.mp4",
     "tmeier1964", "Sea/ocean Atlantic water texture, aerial"),
    ("beach_waves_aerial.mp4",
     "https://cdn.pixabay.com/video/2020/09/23/50638-460909363_small.mp4",
     "daniloriba", "Beach waves ocean aerial texture"),
]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets" / "wind_test_videos"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    credits = []

    for filename, url, author, description in VIDEOS:
        out_path = OUT_DIR / filename
        if not out_path.exists():
            print(f"Downloading {description} -> {out_path}")
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
        credits.append(f"- {filename}: \"{description}\" by {author}, Pixabay License")

    (OUT_DIR / "CREDITS.md").write_text("\n".join(credits) + "\n")
    print(f"\n{len(VIDEOS)} videos ready in {OUT_DIR} (attribution in CREDITS.md)")


if __name__ == "__main__":
    main()
