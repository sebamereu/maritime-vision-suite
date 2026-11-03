import cv2
import numpy as np
from dataset import IGNORE_INDEX, MaSTrDataset, remap_mask


def test_remap_mask_maps_labels_to_contiguous_classes():
    raw = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    remapped = remap_mask(raw)

    assert remapped[0, 0] == 0  # obstacle
    assert remapped[0, 1] == 1  # water
    assert remapped[1, 0] == 2  # sky
    assert remapped[1, 1] == IGNORE_INDEX


def test_dataset_getitem_shapes_and_dtypes(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    frame = np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[:20, :] = 2  # sky
    mask[20:, :] = 1  # water

    cv2.imwrite(str(images_dir / "0001.jpg"), frame)
    cv2.imwrite(str(masks_dir / "0001m.png"), mask)

    dataset = MaSTrDataset(images_dir, masks_dir, ["0001"], size=(16, 24))

    assert len(dataset) == 1
    image_tensor, mask_tensor = dataset[0]

    assert image_tensor.shape == (3, 16, 24)
    assert image_tensor.dtype.is_floating_point
    assert image_tensor.min() >= 0.0 and image_tensor.max() <= 1.0

    assert mask_tensor.shape == (16, 24)
    assert set(mask_tensor.unique().tolist()) <= {1, 2}
