from metrics import box_iou


def test_box_iou_perfect_overlap_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == 1.0


def test_box_iou_no_overlap_is_zero():
    assert box_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_box_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection 5x10=50, union 100+100-50=150
    assert abs(box_iou(a, b) - 50 / 150) < 1e-6
