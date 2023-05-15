# mge-segment-anything

这是 **MegEngine 版本**的 `SAM` 模型, 相关代码是 [torch 版本](https://github.com/facebookresearch/segment-anything/tree/main) SAM 在 MegEngine 中的实现。

SAM 是一个图片分割的基础模型。它可以根据用户输入的 prompts 来为图片生成 mask。用户也可以使用 SAM 为一张图片中的所有物体生成 mask。[这篇论文](https://ai.facebook.com/research/publications/segment-anything/)中有着关于 SAM 模型的更多信息。

## 环境准备

```bash
pip install megengine opencv-python pycocotools matplotlib
```

## 权值下载

有两个方法可以得到 MegEngine-SAM 的权值：

##### 方法一：直接下载

可以从[这里](https://huggingface.co/ccq-mgevii/MegEngine-SAM/tree/main)直接下载 MegEngine-SAM 的权值，下载完成后请存储为 checkpoints/*.pkl`。

`vit_b: `[`VIT-B Model`](https://huggingface.co/ccq-mgevii/MegEngine-SAM/resolve/main/sam_vit_b_01ec64.pkl)

`vit_l: `[`VIT-L Model`](https://huggingface.co/ccq-mgevii/MegEngine-SAM/resolve/main/sam_vit_l_0b3195.pkl)

`vit_h: `[`VIT-H Model`](https://huggingface.co/ccq-mgevii/MegEngine-SAM/resolve/main/sam_vit_h_4b8939.pkl)

##### 方法二：从 Torch 转换

可以下载 [torch weights](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints) 存储为 checkpoints/*.pth`。

然后执行以下代码进行权值转换:

```python
export PYTHONPATH=/path/to/megengine-sam:$PYTHONPATH
python convert_weights.py
```

转换完成后，被转换好的 MegEngine-SAM 权值会被存为 `./checkpoints/*.pkl`。

## 例子

```python
export PYTHONPATH=/path/to/megengine-sam:$PYTHONPATH
python example.py
```

这个例子会为 `images/src` 底下图片的生成 mask，相关结果会被存储到 `images/dst` 底下。

## 使用

MegEngine-SAM 的 api 和原始版本的 [segment-anything](https://github.com/facebookresearch/segment-anything/tree/main) 保持了一致。

所以你可以用下面的代码根据 prompt 为一张图片生成 mask：

```python
from mge_segment_anything import SamPredictor, sam_model_registry
predictor = SamPredictor(
    sam_model_registry["model_name"](checkpoint="<path/to/checkpoint>")
)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

或者为一张图片中所有物体生成 mask：

```python
from mge_segment_anything import SamAutomaticMaskGenerator, sam_model_registry
mask_generator = SamAutomaticMaskGenerator(
    sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
)
masks = mask_generator.generate(<your_image>)
```
