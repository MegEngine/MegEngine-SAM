import os

from mge_segment_anything import sam_model_registry


def convert():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    checkpoints = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
    }
    for model_name, checkpoint in checkpoints.items():
        cp_path = os.path.join(checkpoint_dir, checkpoint)
        if os.path.exists(cp_path):
            _ = sam_model_registry[model_name](checkpoint=cp_path)


if __name__ == "__main__":
    convert()
