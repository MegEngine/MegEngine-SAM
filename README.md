# mge-segment-anything

This code is a megengine version of `segment-anything`, which is transfered from [torch code](https://github.com/facebookresearch/segment-anything/tree/main).

### prepare the environments

```bash
pip install megengine opencv-python pycocotools matplotlib
```

### download weights
Just download torch weights from https://github.com/facebookresearch/segment-anything#model-checkpoints and save as `./checkpoints/*.pth` and run 

### run

```python
python test.py
```

You can see the results in `images/dst`