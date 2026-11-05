import numpy as np
from optical_flow import dense_flow, flow_variability_map


def _checkerboard(height=128, width=128, square=16):
    board = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, square * 2):
        for x in range(0, width, square * 2):
            board[y:y + square, x:x + square] = 255
            board[y + square:y + 2 * square, x + square:x + 2 * square] = 255
    return board


def test_dense_flow_output_shape():
    frame0 = _checkerboard()
    frame1 = np.roll(frame0, shift=3, axis=1)  # uniform horizontal shift

    flow = dense_flow(frame0, frame1)

    assert flow.shape == (128, 128, 2)


def test_variability_map_shape_and_no_nans():
    frame0 = _checkerboard()
    frame1 = np.roll(frame0, shift=3, axis=1)
    flow = dense_flow(frame0, frame1)

    z = flow_variability_map(flow, patch_size=16)

    assert z.shape == (128, 128)
    assert np.all(np.isfinite(z))


def test_zero_flow_gives_zero_map_without_crashing():
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    z = flow_variability_map(flow)

    assert z.shape == (64, 64)
    assert np.all(z == 0)  # no motion anywhere -> std is 0 -> defined fallback
