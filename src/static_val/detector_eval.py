"""COCO evaluation of one EfficientDet checkpoint on a TFRecord, optionally on a subset of images.

This runs inside the environment of the detector repository
(uncertainty-detection-autolabeling), whose ``src`` modules it imports. It
follows that repository's ``eval.py`` and adds evaluation restricted to a
list of image source ids. Results are written as JSON.

    python -m static_val.detector_eval --autolabeling-dir DIR --hparams YAML \
        --checkpoint DIR --tfrecord FILE --num-images N [--image-ids FILE] --output FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BATCH_SIZE = 8
MODEL_NAME = "efficientdet-d0"


def evaluate(
    autolabeling_dir: Path,
    hparams: Path,
    checkpoint: Path,
    tfrecord: Path,
    num_images: int,
    image_ids: list[int] | None,
) -> dict[str, float]:
    sys.path.insert(0, str(autolabeling_dir / "src"))
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import anchors
    import coco_metric
    import dataloader
    import efficientdet_keras
    import hparams_config
    import label_util
    import postprocess
    import tensorflow as tf
    import utils
    import utils_keras
    from absl import logging
    from utils_extra import mc_eval

    logging.set_verbosity(logging.ERROR)
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    config = hparams_config.get_efficientdet_config(MODEL_NAME)
    config.override(str(hparams))
    config.val_json_file = None
    config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
    config.drop_remainder = False
    config.image_size = utils.parse_image_size(config["image_size"])

    if tf.config.list_physical_devices("GPU"):
        strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")

    label_map = label_util.get_label_map(config.label_map)
    evaluator = coco_metric.EvaluationMetric(
        filename=config.val_json_file, label_map=label_map, apiou_curve=False
    )

    with strategy.scope():
        model = efficientdet_keras.EfficientDetNet(config=config)
        model.build((None, *config.image_size, 3))
        utils_keras.restore_ckpt(
            model,
            tf.train.latest_checkpoint(str(checkpoint)),
            config.moving_average_decay,
            skip_mismatch=False,
        )

        @tf.function
        def model_fn(images, labels):
            if config.mc_dropout:
                cls_outputs, box_outputs = mc_eval(model, images, config)
            else:
                cls_outputs, box_outputs = model(images, training=False)
            detections = postprocess.generate_detections(
                config, cls_outputs, box_outputs, labels["image_scales"], labels["source_ids"]
            )
            detections = postprocess.transform_detections(detections)
            tf.numpy_function(evaluator.update_state, [labels["groundtruth_data"], detections], [])

        reader = dataloader.InputReader(
            str(tfrecord),
            is_training=False,
            max_instances_per_image=config.max_instances_per_image,
        )
        if image_ids is None:
            ds = reader(config, batch_size=BATCH_SIZE)
            ds = ds.take((num_images + BATCH_SIZE - 1) // BATCH_SIZE)
        else:
            # Single-image batches so every record can be kept or dropped by its source id.
            allowed = tf.constant(image_ids, dtype=tf.int64)
            ds = reader(config, batch_size=1).take(num_images)
            ds = ds.filter(
                lambda images, labels: tf.reduce_any(
                    tf.equal(tf.cast(labels["source_ids"][0], tf.int64), allowed)
                )
            )
            ds = ds.unbatch().batch(BATCH_SIZE, drop_remainder=False)

        for images, labels in strategy.experimental_distribute_dataset(ds):
            strategy.run(model_fn, (images, labels))

    values = evaluator.result()
    metrics = {name: float(values[i]) for i, name in enumerate(evaluator.metric_names)}
    if label_map:
        offset = len(evaluator.metric_names)
        for i, cid in enumerate(sorted(label_map)):
            metrics[f"AP_/{label_map[cid]}"] = float(values[offset + i])
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--autolabeling-dir", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tfrecord", type=Path, required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--image-ids", type=Path, help="Text file, one source id per line.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    image_ids = None
    if args.image_ids is not None:
        with open(args.image_ids) as f:
            image_ids = [int(line) for line in f if line.strip()]

    metrics = evaluate(
        args.autolabeling_dir,
        args.hparams,
        args.checkpoint,
        args.tfrecord,
        args.num_images,
        image_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
