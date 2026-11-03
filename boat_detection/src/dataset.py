"""Pascal VOC XML -> single-class YOLO label conversion for SeaShips.

SeaShips(7000) (Shao et al., IEEE TMM 2018) ships six ship-type classes
(bulk cargo carrier, container ship, fishing boat, general cargo ship,
ore carrier, passenger ship). This project collapses all of them to one
class, "boat": the roadmap goal is detecting that a boat is there, not
classifying ship type, and six classes over a few hundred training
images would mostly measure hull memorization rather than detection.
"""
from xml.etree import ElementTree as ET

CLASS_NAMES = ("boat",)
NUM_CLASSES = 1


def parse_voc_boxes(xml_path):
    """Reads one VOC annotation file.

    Returns (image_width, image_height, boxes), boxes a list of
    (xmin, ymin, xmax, ymax) in absolute pixel coordinates, one per
    <object>. The original per-object <name> (ship type) is read but
    discarded -- every box becomes class 0, "boat".
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))

    return width, height, boxes


def boxes_to_yolo_lines(width, height, boxes, class_id=0):
    """Converts absolute-pixel boxes to YOLO label lines
    ("class_id cx cy w h", normalized to [0, 1]).

    class_id defaults to 0 (the fine-tuning label set, where "boat" is
    the only class). evaluate_boat_detection.py also needs a *second*
    label set with class_id=COCO_BOAT_CLASS_ID, so that ultralytics'
    own val() can score the zero-shot pretrained model (whose own
    class indices are COCO's, not this project's) with its standard
    metric instead of a hand-rolled one.
    """
    lines = []
    for xmin, ymin, xmax, ymax in boxes:
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def write_yolo_label(xml_path, dest_path, class_id=0):
    """Reads one VOC annotation and writes its YOLO-format .txt label.
    Returns the number of boxes written.
    """
    width, height, boxes = parse_voc_boxes(xml_path)
    lines = boxes_to_yolo_lines(width, height, boxes, class_id=class_id)
    dest_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def yolo_lines_to_boxes(lines, width, height):
    """Inverse of boxes_to_yolo_lines: parses YOLO label lines back to
    absolute-pixel (xmin, ymin, xmax, ymax) boxes. Class id is ignored,
    there's only one class.
    """
    boxes = []
    for line in lines:
        _, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes
