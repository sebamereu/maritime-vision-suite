import config


def test_video_path_points_at_data_videos_sample():
    assert config.VIDEO_PATH.parent.name == "videos"
    assert config.VIDEO_PATH.name == "sample.mp4"


def test_project_root_is_the_repo_root():
    assert (config.PROJECT_ROOT / "requirements.txt").exists()
    assert (config.PROJECT_ROOT / "README.md").exists()
