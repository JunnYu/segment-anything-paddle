# TODO
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import paddle
# import paddle.nn as nn

# from segment_anything_paddle import sam_model_registry
# from segment_anything_paddle.utils.fd_model import SamFastDeployModel

# import argparse

# try:
#     import fastdeploy  # type: ignore

#     fastdeploy_exists = True
# except ImportError:
#     fastdeploy_exists = False

# parser = argparse.ArgumentParser(
#     description="Export the SAM prompt encoder and mask decoder to an FastDeploy model."
# )

# parser.add_argument(
#     "--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint."
# )

# parser.add_argument(
#     "--output", type=str, required=True, help="The filename to save the FastDeploy model to."
# )

# parser.add_argument(
#     "--model-type",
#     type=str,
#     required=True,
#     help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
# )

# parser.add_argument(
#     "--return-single-mask",
#     action="store_true",
#     help=(
#         "If true, the exported ONNX model will only return the best mask, "
#         "instead of returning multiple masks. For high resolution images "
#         "this can improve runtime when upscaling masks is expensive."
#     ),
# )

# parser.add_argument(
#     "--gelu-approximate",
#     action="store_true",
#     help=(
#         "Replace GELU operations with approximations using tanh. Useful "
#         "for some runtimes that have slow or unimplemented erf ops, used in GELU."
#     ),
# )

# parser.add_argument(
#     "--use-stability-score",
#     action="store_true",
#     help=(
#         "Replaces the model's predicted mask quality score with the stability "
#         "score calculated on the low resolution masks using an offset of 1.0. "
#     ),
# )

# parser.add_argument(
#     "--return-extra-metrics",
#     action="store_true",
#     help=(
#         "The model will return five results: (masks, scores, stability_scores, "
#         "areas, low_res_logits) instead of the usual three. This can be "
#         "significantly slower for high resolution outputs."
#     ),
# )


# def run_export(
#     model_type: str,
#     checkpoint: str,
#     output: str,
#     return_single_mask: bool,
#     gelu_approximate: bool = False,
#     use_stability_score: bool = False,
#     return_extra_metrics=False,
# ):
#     print("Loading model...")
#     sam = sam_model_registry[model_type](checkpoint=checkpoint)

#     fastdeploy_model = SamFastDeployModel(
#         model=sam,
#         return_single_mask=return_single_mask,
#         use_stability_score=use_stability_score,
#         return_extra_metrics=return_extra_metrics,
#     )

#     if gelu_approximate:
#         for n, m in fastdeploy_model.named_sublayers(include_self=True):
#             if isinstance(m, nn.GELU):
#                 m._approximate = True

#     dynamic_axes = {
#         "point_coords": {1: "num_points"},
#         "point_labels": {1: "num_points"},
#     }

#     embed_dim = sam.prompt_encoder.embed_dim
#     embed_size = sam.prompt_encoder.image_embedding_size
#     mask_input_size = [4 * x for x in embed_size]
#     dummy_inputs = {
#         "image_embeddings": paddle.randn([1, embed_dim, *embed_size], dtype=paddle.float32),
#         "point_coords": paddle.randint(low=0, high=1024, shape=(1, 5, 2), dtype=paddle.float32),
#         "point_labels": paddle.randint(low=0, high=4, shape=(1, 5), dtype=paddle.float32),
#         "mask_input": paddle.randn([1, 1, *mask_input_size], dtype=paddle.float32),
#         "has_mask_input": paddle.to_tensor([1], dtype=paddle.float32),
#         "orig_im_size": paddle.to_tensor([1500, 2250], dtype=paddle.float32),
#     }

#     _ = fastdeploy_model(**dummy_inputs)

#     output_names = ["masks", "iou_predictions", "low_res_masks"]

#     if fastdeploy_exists:
#         fd_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
#         fd_session = InferenceSession(output)
#         _ = fd_session.run(None, fd_inputs)
#         print("Model has successfully been run with FastDeploy.")


# def to_numpy(tensor):
#     return tensor.cpu().numpy()


# if __name__ == "__main__":
#     args = parser.parse_args()
#     run_export(
#         model_type=args.model_type,
#         checkpoint=args.checkpoint,
#         output=args.output,
#         return_single_mask=args.return_single_mask,
#         gelu_approximate=args.gelu_approximate,
#         use_stability_score=args.use_stability_score,
#         return_extra_metrics=args.return_extra_metrics,
#     )