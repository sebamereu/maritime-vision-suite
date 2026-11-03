import pytest
from video_loader import open_video


def test_open_video_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(IOError):
        open_video(str(missing))
