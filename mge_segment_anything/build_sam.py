import os
from functools import partial

import numpy as np

import megengine as mge
from megengine import module as M

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)


def build_sam_vit_h(checkpoint=None, load_from_torch_weights=True):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        load_from_torch_weights=load_from_torch_weights,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, load_from_torch_weights=True):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        load_from_torch_weights=load_from_torch_weights,
    )


def build_sam_vit_b(checkpoint=None, load_from_torch_weights=True):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        load_from_torch_weights=load_from_torch_weights,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    load_from_torch_weights=False,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(M.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    mge_state_dict = sam.state_dict()
    if checkpoint is not None:
        if load_from_torch_weights:

            def get_state_dict_from_torch(checkpoint):
                import torch

                with open(checkpoint, "rb") as f:
                    torch_state_dict = torch.load(f)

                either_have = set(mge_state_dict.keys()) | set(torch_state_dict.keys())
                both_have = set(mge_state_dict.keys()) & set(torch_state_dict.keys())
                only_mge_have = set(mge_state_dict.keys()) - set(
                    torch_state_dict.keys()
                )
                only_torch_have = set(torch_state_dict.keys()) - set(
                    mge_state_dict.keys()
                )

                if not either_have == both_have:
                    if len(only_mge_have) != 0:
                        print(f"mge have but torch miss:")
                        for k in only_mge_have:
                            print(f"    {k}: {mge_state_dict[k].shape}")
                    if len(only_torch_have) != 0:
                        print(f"torch have but mge miss:")
                        for k in only_torch_have:
                            print(f"    {k}: {torch_state_dict[k].shape}")

                processed_state_dict = {}
                for k in both_have:
                    mv = mge_state_dict[k]
                    tv = torch_state_dict[k]
                    if mv.shape != tuple(tv.shape):
                        if np.prod(mv.shape) != np.prod(tv.shape):
                            print(
                                f"{k}: mge-shape{mv.shape}, torch-shape{tuple(tv.shape)}"
                            )
                            continue
                        tv = tv.reshape(mv.shape)
                    processed_state_dict[k] = np.array(tv.cpu().numpy(), dtype=mv.dtype)
                return processed_state_dict

            mge_state_dict = get_state_dict_from_torch(checkpoint)
            sam.load_state_dict(mge_state_dict, strict=False)
            if not os.path.exists(checkpoint.replace(".pth", ".pkl")):
                with open(checkpoint.replace(".pth", ".pkl"), "wb") as f:
                    mge.save(sam.state_dict(), f)
        else:
            with open(checkpoint, "rb") as f:
                mge_state_dict = mge.load(f)
            sam.load_state_dict(mge_state_dict)

    return sam
