import os
import time

import cv2
import numpy as np

import matplotlib.pyplot as plt
import megengine
from megengine.utils.profiler import Profiler
from mge_segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)


def perf_impl(hint, workload, warm_iter, run_iter, sync_func=megengine._full_sync):
    for _ in range(warm_iter):
        workload()

    sync_func()
    start = time.perf_counter()
    for _ in range(run_iter):
        workload()
    sync_func()
    end = time.perf_counter()

    print(
        f"{hint}: {end - start} s for {run_iter} iters, {(end - start) / run_iter} s/iter"
    )


def predictor():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    sam = sam_model_registry["vit_b"](
        checkpoint=os.path.join(checkpoint_dir, "sam_vit_b_01ec64.pth")
    )
    predictor = SamPredictor(sam)

    img_path = os.path.join(os.path.dirname(__file__), "images", "src", "dog.jpg")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    point_coords = np.array([[img.shape[0] // 2, img.shape[1] // 2]])
    point_labels = np.array([1])

    def workload():
        predictor.set_image(img)
        masks, _, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=True,
        )

    return workload


def automatic_mask_generator():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    sam = sam_model_registry["vit_b"](
        checkpoint=os.path.join(checkpoint_dir, "sam_vit_b_01ec64.pth")
    )
    mask_generator = SamAutomaticMaskGenerator(sam)

    img_path = os.path.join(os.path.dirname(__file__), "images", "src", "dog.jpg")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    def workload():
        masks = mask_generator.generate(img)

    return workload


def perf_predictor(warm_iter, run_iter):
    perf_impl("predictor", predictor(), warm_iter, run_iter)


def perf_automatic_mask_generator(warm_iter, run_iter):
    perf_impl(
        "automatic_mask_generator", automatic_mask_generator(), warm_iter, run_iter
    )


if __name__ == "__main__":
    # perf_predictor(10, 100)
    perf_automatic_mask_generator(2, 5)
    # workload = automatic_mask_generator()
    # for _ in range(2):
    #     workload()
    # megengine._full_sync()

    # profiler = Profiler(with_backtrace=False)
    # with profiler:
    #     for _ in range(1):
    #         workload()
