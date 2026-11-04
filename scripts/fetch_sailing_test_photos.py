#!/usr/bin/env python3
"""Download a small set of free-license sailboat photos from Unsplash
(Unsplash License: free for commercial and non-commercial use, no
permission required) for the domain-generalization check in
test_on_real_sailing_photos.py.

This is the companion to fetch_ac75_test_photos.py: classic sailboats
instead of foiling AC75s, but the same point -- SeaShips has zero
sailboats of any kind, so this is the real test of what fine-tuning on
it does to the model's ability to recognize them.

Usage:
    python scripts/fetch_sailing_test_photos.py
"""
import urllib.request
from pathlib import Path

PHOTOS = [
    ("sailing_test_1.jpg", "https://images.unsplash.com/photo-1785716744224-1f428b7d05ca?fm=jpg&q=80&w=1600",
     "Lee Shepherd", "Three sailboats navigate choppy blue water with whitecaps"),
    ("sailing_test_2.jpg", "https://images.unsplash.com/photo-1613578699399-82ae71be53a3?fm=jpg&q=80&w=1600",
     "Unsplash", "White sail boat on sea during daytime"),
    ("sailing_test_3.jpg", "https://images.unsplash.com/photo-1506527240747-720a3e17b910?fm=jpg&q=80&w=1600",
     "Unsplash", "Man riding sailboat"),
    ("sailing_test_4.jpg", "https://images.unsplash.com/photo-1501771924607-209f42a6e7e4?fm=jpg&q=80&w=1600",
     "Unsplash", "Three sail boats on water during daytime"),
]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets" / "sailing_test_photos"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    credits = []

    for filename, url, author, description in PHOTOS:
        out_path = OUT_DIR / filename
        if not out_path.exists():
            print(f"Downloading {description} -> {out_path}")
            urllib.request.urlretrieve(url, out_path)
        credits.append(f"- {filename}: \"{description}\" by {author}, Unsplash License")

    (OUT_DIR / "CREDITS.md").write_text("\n".join(credits) + "\n")
    print(f"\n{len(PHOTOS)} photos ready in {OUT_DIR} (attribution in CREDITS.md)")


if __name__ == "__main__":
    main()
