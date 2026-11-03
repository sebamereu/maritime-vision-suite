from dataset import (
    boxes_to_yolo_lines,
    parse_voc_boxes,
    write_yolo_label,
    yolo_lines_to_boxes,
)

VOC_XML = """<annotation>
    <size>
        <width>100</width>
        <height>50</height>
    </size>
    <object>
        <name>fishing boat</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>5</ymin>
            <xmax>30</xmax>
            <ymax>25</ymax>
        </bndbox>
    </object>
    <object>
        <name>container ship</name>
        <bndbox>
            <xmin>50</xmin>
            <ymin>10</ymin>
            <xmax>90</xmax>
            <ymax>40</ymax>
        </bndbox>
    </object>
</annotation>
"""


def test_parse_voc_boxes_reads_size_and_boxes_ignoring_ship_type(tmp_path):
    xml_path = tmp_path / "000001.xml"
    xml_path.write_text(VOC_XML)

    width, height, boxes = parse_voc_boxes(xml_path)

    assert (width, height) == (100, 50)
    assert boxes == [(10.0, 5.0, 30.0, 25.0), (50.0, 10.0, 90.0, 40.0)]


def test_boxes_to_yolo_lines_normalizes_and_collapses_to_class_zero():
    lines = boxes_to_yolo_lines(100, 50, [(10.0, 5.0, 30.0, 25.0)])

    assert lines == ["0 0.200000 0.300000 0.200000 0.400000"]


def test_boxes_to_yolo_lines_empty_boxes_returns_empty_list():
    assert boxes_to_yolo_lines(100, 50, []) == []


def test_write_yolo_label_writes_one_line_per_box(tmp_path):
    xml_path = tmp_path / "000001.xml"
    xml_path.write_text(VOC_XML)
    dest_path = tmp_path / "000001.txt"

    n = write_yolo_label(xml_path, dest_path)

    assert n == 2
    lines = dest_path.read_text().splitlines()
    assert lines == [
        "0 0.200000 0.300000 0.200000 0.400000",
        "0 0.700000 0.500000 0.400000 0.600000",
    ]


def test_boxes_to_yolo_lines_accepts_a_different_class_id():
    lines = boxes_to_yolo_lines(100, 50, [(10.0, 5.0, 30.0, 25.0)], class_id=8)

    assert lines == ["8 0.200000 0.300000 0.200000 0.400000"]


def test_yolo_lines_to_boxes_is_the_inverse_of_boxes_to_yolo_lines():
    original = [(10.0, 5.0, 30.0, 25.0), (50.0, 10.0, 90.0, 40.0)]
    lines = boxes_to_yolo_lines(100, 50, original)

    recovered = yolo_lines_to_boxes(lines, 100, 50)

    for (ox0, oy0, ox1, oy1), (rx0, ry0, rx1, ry1) in zip(original, recovered):
        assert abs(ox0 - rx0) < 1e-3
        assert abs(oy0 - ry0) < 1e-3
        assert abs(ox1 - rx1) < 1e-3
        assert abs(oy1 - ry1) < 1e-3
