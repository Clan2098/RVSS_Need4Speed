LABELS = [
    "uleft",
    "sharpleft",
    "left",
    "straight",
    "right",
    "sharpright",
    "uright"
]

LABEL_TO_ANGLE = {
    0: -1,
    1: -0.4,
    2: -0.2,
    3: 0.0,
    4: 0.2,
    5: 0.4,
    6: 1
}


def steering_to_class(steering):
    if steering <= -0.4:
        return 0
    if steering <= -0.2:
        return 1
    if steering < 0:
        return 2
    if steering == 0:
        return 3
    if steering <= 0.2:
        return 4
    if steering <= 0.4:
        return 5

    return 6


def label_to_class(label):
    """Convert steering type label string to class number."""
    label = label.replace("-","")
    if label in LABELS:
        return LABELS.index(label)
    raise ValueError(f"Unknown steering label: {label}")


def label_to_angle(label):
    """Convert steering type label string to steering angle."""
    class_num = label_to_class(label)
    return LABEL_TO_ANGLE[class_num]
