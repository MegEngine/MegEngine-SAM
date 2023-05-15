import os
import shutil

import cv2
import numpy as np

import matplotlib.pyplot as plt
from mge_segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)

checkpoints = {
    "vit_b": "sam_vit_b_01ec64.pkl",
    "vit_h": "sam_vit_h_4b8939.pkl",
    "vit_l": "sam_vit_l_0b3195.pkl",
}


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def test_predictor(model_name="vit_b"):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    sam = sam_model_registry[model_name](
        checkpoint=os.path.join(checkpoint_dir, checkpoints[model_name])
    )
    predictor = SamPredictor(sam)

    src_img_dir = os.path.join(os.path.dirname(__file__), "images", "src")
    dst_img_dir = os.path.join(os.path.dirname(__file__), "images", "dst")
    if os.path.exists(dst_img_dir):
        shutil.rmtree(dst_img_dir)
    os.mkdir(dst_img_dir)

    for img_file in sorted(os.listdir(src_img_dir)):
        img_path = os.path.join(src_img_dir, img_file)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print("*" * 10, img_path, "*" * 10)
        predictor.set_image(img)
        masks, _, _ = predictor.predict(
            point_coords=np.array([[img.shape[0] // 2, img.shape[1] // 2]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        for i, mask in enumerate(masks):
            mask = show_mask(mask, plt.gca(), True)
            plt.savefig(
                os.path.join(dst_img_dir, f"{img_file.split('.')[0]}_mask_{i}.png")
            )


def test_automatic_mask_generator(model_name="vit_b"):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    sam = sam_model_registry[model_name](
        checkpoint=os.path.join(checkpoint_dir, checkpoints[model_name])
    )
    mask_generator = SamAutomaticMaskGenerator(sam)

    src_img_dir = os.path.join(os.path.dirname(__file__), "images", "src")
    dst_img_dir = os.path.join(os.path.dirname(__file__), "images", "dst")
    if os.path.exists(dst_img_dir):
        shutil.rmtree(dst_img_dir)
    os.mkdir(dst_img_dir)

    for img_file in sorted(os.listdir(src_img_dir)):
        img_path = os.path.join(src_img_dir, img_file)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print("*" * 10, img_path, "*" * 10)
        masks = mask_generator.generate(img)
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks)

        plt.savefig(os.path.join(dst_img_dir, f"{img_file.split('.')[0]}.png"))


if __name__ == "__main__":
    test_automatic_mask_generator("vit_h")
