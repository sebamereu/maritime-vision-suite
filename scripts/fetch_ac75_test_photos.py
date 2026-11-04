#!/usr/bin/env python3
"""Download a small set of real AC75/foiling America's Cup photos from
Wikimedia Commons (Creative Commons licensed, with attribution) for the
domain-generalization check in test_on_real_sailing_photos.py.

Not a training set -- genuinely free-licensed AC75 imagery is scarce
(the AC75 and Ineos Britannia Commons categories hold under 15 files
combined; see README). This is a qualitative test set only: can a
boat-detection model recognize a foiling hull with no hull in the
water, which looks nothing like the commercial ships it may have been
trained on.

Wikimedia requires a descriptive User-Agent per their robot policy
(https://w.wiki/4wJS) -- requests without one are rejected with a 429.

Usage:
    python scripts/fetch_ac75_test_photos.py
"""
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "maritime-vision-suite-research/1.0 (portfolio project; contact: sebamere00@gmail.com)"

# (output filename, Commons file title, photographer, license)
# All from Category:Ineos Britannia / Category:Team New Zealand on
# Wikimedia Commons; see each file's page for the full license text.
PHOTOS = [
    ("ac75_1_dji.jpg", "C.GREGORY-INEOS BRITANNIA 20240516 DJI 0719.jpg",
     "C. Gregory", "CC BY-SA 4.0"),
    ("ac75_2_ineos.jpg", "Ineos Team UK AC75.png",
     "Geoff McKay", "CC BY 2.0"),
    ("ac75_3_startbox.jpg", "Prada Cup Final Race One - Entering the Start Box (50951086996).jpg",
     "Ricardo Pinto / America's Cup Media", "CC BY-SA 2.0"),
    ("ac75_4_upwind.jpg", "Prada Cup Final Race One - First Upwind Leg! (50951184492).jpg",
     "Ricardo Pinto / America's Cup Media", "CC BY-SA 2.0"),
    ("ac75_5_offfoils.jpg", "Prada Cup Final Race One - INEOS Off it's Foils! (50951181827).jpg",
     "Ricardo Pinto / America's Cup Media", "CC BY-SA 2.0"),
    ("ac75_6_ineosteam.jpg", "Prada Cup Final Race One - INEOS TEAM UK (50951179977).jpg",
     "Ricardo Pinto / America's Cup Media", "CC BY-SA 2.0"),
    ("ac75_7_lunarossa.jpg", "Prada Cup Final Race One - Luna Rossa Sails past INEOS! (50951088761).jpg",
     "Ricardo Pinto / America's Cup Media", "CC BY-SA 2.0"),
    ("ac75_8_tacking.jpg", "Prada Cup Final Race One - Tacking Upwind! (50951085726).jpg",
     "Ricardo Pinto / America's Cup Media", "CC BY-SA 2.0"),
    ("ac75_9_nz_foiling.jpg", "Emirates Team New Zealand Foiling 8-21-13 - panoramio.jpg",
     "ehmueller (Panoramio)", "CC BY 3.0"),
    ("ac75_10_lunarossa_vs_nz.jpg", "Luna Rossa Italy vs. Emirates Team New Zealand - panoramio.jpg",
     "Erik Charlton (Panoramio)", "CC BY 3.0"),
    ("ac75_11_nz_alinghi.jpg", "Teamnz with alinghi in background.jpg",
     "Steffen Flor", "CC BY-SA 3.0"),
]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets" / "ac75_test_photos"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    credits = []

    for filename, commons_title, author, license_name in PHOTOS:
        out_path = OUT_DIR / filename
        encoded = urllib.parse.quote(commons_title)
        url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}?width=1200"

        if not out_path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            print(f"Downloading {commons_title} -> {out_path}")
            with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
                f.write(resp.read())

        credits.append(
            f"- {filename}: \"{commons_title}\" by {author}, {license_name}, "
            f"via Wikimedia Commons"
        )

    (OUT_DIR / "CREDITS.md").write_text("\n".join(credits) + "\n")
    print(f"\n{len(PHOTOS)} photos ready in {OUT_DIR} (attribution in CREDITS.md)")


if __name__ == "__main__":
    main()
