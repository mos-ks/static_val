"""BDD100K validation TFRecord in KITTI class ids, restricted to the classes both share.

BDD ``car``, ``truck`` and ``pedestrian`` keep their KITTI ids. ``rider`` is
written as KITTI ``cyclist``. Images without any of these classes are
skipped. Source ids run from 0 in file order. Requires TensorFlow, ijson and
Pillow (the detector environment).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

BDD_TO_KITTI = {"car": (1, "car"), "truck": (3, "truck"), "pedestrian": (4, "pedestrian"),
                "rider": (6, "cyclist")}  # fmt: skip


def write_bdd_tfrecord(labels_json: Path, image_dir: Path, output: Path) -> int:
    import ijson
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    def int64(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def floats(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def bytes_(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with tf.io.TFRecordWriter(str(output)) as writer, open(labels_json, "rb") as f:
        for item in ijson.items(f, "item"):
            image_path = image_dir / item["name"]
            if "labels" not in item or not image_path.is_file():
                continue
            boxes, names, ids = [], [], []
            for label in item["labels"]:
                mapped = BDD_TO_KITTI.get(label["category"])
                box = label.get("box2d")
                if mapped is None or box is None:
                    continue
                boxes.append([float(box[k]) for k in ("x1", "y1", "x2", "y2")])
                ids.append(mapped[0])
                names.append(mapped[1].encode())
            if not boxes:
                continue

            encoded = image_path.read_bytes()
            h, w = np.asarray(Image.open(io.BytesIO(encoded))).shape[:2]
            b = np.asarray(boxes)
            feature = {
                "image/height": int64([h]),
                "image/width": int64([w]),
                "image/filename": bytes_([item["name"].encode()]),
                "image/source_id": bytes_([str(written).encode()]),
                "image/key/sha256": bytes_([hashlib.sha256(encoded).hexdigest().encode()]),
                "image/encoded": bytes_([encoded]),
                "image/format": bytes_([b"jpg"]),
                "image/object/bbox/xmin": floats((b[:, 0] / w).tolist()),
                "image/object/bbox/xmax": floats((b[:, 2] / w).tolist()),
                "image/object/bbox/ymin": floats((b[:, 1] / h).tolist()),
                "image/object/bbox/ymax": floats((b[:, 3] / h).tolist()),
                "image/object/class/text": bytes_(names),
                "image/object/class/label": int64(ids),
                "image/object/difficult": int64([0] * len(ids)),
                "image/object/truncated": int64([0] * len(ids)),
                "image/object/occluded": int64([0] * len(ids)),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            written += 1
    return written
