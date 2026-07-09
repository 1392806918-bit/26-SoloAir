COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

_NORMALIZED_TO_NAME = {}
for _name in COCO_CLASSES:
    _NORMALIZED_TO_NAME[" ".join(_name.lower().replace("_", " ").replace("-", " ").split())] = _name

TARGET_ALIASES = {
    "sportsball": "sports ball",
    "sports_ball": "sports ball",
    "sports-ball": "sports ball",
    "ball": "sports ball",
}


def normalize_target_name(value):
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def class_name_for_id(class_id):
    class_id = int(class_id)
    if class_id < 0 or class_id >= len(COCO_CLASSES):
        raise ValueError(f"COCO class_id out of range: {class_id}")
    return COCO_CLASSES[class_id]


def resolve_target_class(target=None, class_id=None):
    if target is not None:
        normalized = normalize_target_name(target)
        normalized = TARGET_ALIASES.get(normalized, normalized)
        if normalized not in _NORMALIZED_TO_NAME:
            raise ValueError(f"unsupported COCO target: {target}")
        name = _NORMALIZED_TO_NAME[normalized]
        return {"class_id": COCO_CLASSES.index(name), "name": name}

    if class_id is None:
        raise ValueError("target or class_id is required")

    class_id = int(class_id)
    return {"class_id": class_id, "name": class_name_for_id(class_id)}
