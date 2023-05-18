# mge-segment-anything

[中文 ReadMe](./README_cn.md)

This code is a **megengine version** of `Segment Anything Model (SAM)`, which is transfered from [torch code](https://github.com/facebookresearch/segment-anything/tree/main).

The Segment Anything Model (SAM) is a foundation model for image segmentation. It can generate masks according to the input prompts such as points or boxes. User can use it to generate masks for all objects in an image. For more information of SAM, you can reference this [paper](https://ai.facebook.com/research/publications/segment-anything/).

## prepare the environments

```bash
pip install megengine opencv-python pycocotools matplotlib
```

## Download weights

There are two ways to get the MegEngine-SAM weights:

##### Way1: download directly

You can download MegEngine-SAM weights from [here](https://huggingface.co/ccq-mgevii/MegEngine-SAM/tree/main) and save as `./checkpoints/*.pkl`.

`vit_b: `[`VIT-B Model`](https://huggingface.co/ccq-mgevii/MegEngine-SAM/resolve/main/sam_vit_b_01ec64.pkl)

`vit_l: `[`VIT-L Model`](https://huggingface.co/ccq-mgevii/MegEngine-SAM/resolve/main/sam_vit_l_0b3195.pkl)

`vit_h: `[`VIT-H Model`](https://huggingface.co/ccq-mgevii/MegEngine-SAM/resolve/main/sam_vit_h_4b8939.pkl)

##### Way2: convert from torch weights

You can download [torch weights](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints) and save as `./checkpoints/*.pth`.

Then run:

```python
export PYTHONPATH=/path/to/megengine-sam:$PYTHONPATH
python convert_weights.py
```

The converted MegEngine-SAM weights is saved as `./checkpoints/*.pkl`.

## Example

```python
export PYTHONPATH=/path/to/megengine-sam:$PYTHONPATH
python example.py
```

This example can generate masks for the images in `images/src`, and the results are saved in `images/dst`.

## Usage

MegEngine-SAM have the same api as [segment-anything](https://github.com/facebookresearch/segment-anything/tree/main).

So you can use MegEngine-SAM to generate mask with the prompt like the torch version:

```python
from mge_segment_anything import SamPredictor, sam_model_registry

predictor = SamPredictor(pretrained=True)

# If you are unable to download the weights due to network issues
# you can manually load the downloaded weights using the following method.

predictor = SamPredictor(
    sam_model_registry["model_name"](checkpoint="<path/to/checkpoint>")
)

predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```


Or generate masks for a whole image:

```python
from mge_segment_anything import SamAutomaticMaskGenerator, sam_model_registry

mask_generator = SamAutomaticMaskGenerator(
    sam_model_registry["<model_type>"](pretrained=True)
)

# If you are unable to download the weights due to network issues
# you can manually load the downloaded weights using the following method.

mask_generator = SamAutomaticMaskGenerator(
    sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
)
masks = mask_generator.generate(<your_image>)
```
