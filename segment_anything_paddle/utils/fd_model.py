# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import paddle

import paddle.nn as nn
from paddle.nn import functional as F

from typing import Tuple

from ..modeling import Sam
from .amg import calculate_stability_score


class SamFastDeployModel(nn.Layer):
    """
    This model should not be called directly, but is used in FastDeploy export.
    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the FastDeploy export script for details.
    """

    def __init__(
        self,
        model: Sam,
        return_single_mask: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    @staticmethod
    def resize_longest_image_size(
        input_image_size: paddle.Tensor, longest_side: int
    ) -> paddle.Tensor:
        input_image_size = input_image_size.cast(paddle.float32)
        scale = longest_side / paddle.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = paddle.floor(transformed_size + 0.5).cast(paddle.int64)
        return transformed_size

    def _embed_points(self, point_coords: paddle.Tensor, point_labels: paddle.Tensor) -> paddle.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1).cast(point_embedding.dtype)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: paddle.Tensor, has_mask_input: paddle.Tensor) -> paddle.Tensor:
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape([1, -1, 1, 1])
        return mask_embedding

    def mask_postprocessing(self, masks: paddle.Tensor, orig_im_size: paddle.Tensor) -> paddle.Tensor:
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size)
        masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]

        orig_im_size = orig_im_size.cast(paddle.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks

    def select_masks(
        self, masks: paddle.Tensor, iou_preds: paddle.Tensor, num_points: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = paddle.to_tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        )
        score = iou_preds + (num_points - 2.5) * score_reweight

        # TODO junnyu
        best_idx = paddle.argmax(score, axis=1, dtype="int32")
        bs_range = paddle.arange(masks.shape[0], dtype="int32")
        masks = masks.gather_nd(
            paddle.stack([bs_range, best_idx], axis=-1)
        ).unsqueeze(1)
        iou_preds = iou_preds.gather_nd(
            paddle.stack([bs_range, best_idx], axis=-1)
        ).unsqueeze(1)
        # masks = masks[paddle.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        # iou_preds = iou_preds[paddle.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @paddle.no_grad()
    def forward(
        self,
        image_embeddings: paddle.Tensor,
        point_coords: paddle.Tensor,
        point_labels: paddle.Tensor,
        mask_input: paddle.Tensor,
        has_mask_input: paddle.Tensor,
        orig_im_size: paddle.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = self.mask_postprocessing(masks, orig_im_size)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(
                upscaled_masks, self.model.mask_threshold, self.stability_score_offset
            )
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, 