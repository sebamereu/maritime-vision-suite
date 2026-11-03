import numpy as np
from model import COCO_BOAT_CLASS_ID, BoatDetector

# Instantiating YOLO("yolov8n.pt") downloads the ~6MB nano checkpoint from
# ultralytics on first run (cached afterwards) -- unlike the segmentation
# tests, there's no way to exercise the real wrapper without it.


def test_coco_boat_class_id_matches_the_pretrained_names():
    detector = BoatDetector()

    assert detector.model.names[COCO_BOAT_CLASS_ID] == "boat"


def test_predict_runs_on_a_single_synthetic_image():
    detector = BoatDetector()
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    results = detector.predict(image)

    assert len(results) == 1
    assert hasattr(results[0], "boxes")
