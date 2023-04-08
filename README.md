# Segment Anything With PaddlePaddle
> Notes: This is the code implemented with the PaddlePaddle framework by @junnyu.

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Installation

The code requires `python>=3.7`, as well as `paddlepaddle-gpu>=2.4.2`. Please follow the instructions [here](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) to install Paddle dependencies. Installing both PaddlePaddle with CUDA support is strongly recommended.

If you want to use `paddlepaddle xformers`, you need to install `develop` version `PaddlePaddle` by using cmd `python -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html`.

Install Segment Anything:

```sh
pip install git+https://github.com/junnyu/segment-anything-paddle.git
```

or clone the repository locally and install with

```sh
git clone https://github.com/junnyu/segment-anything-paddle.git
cd segment-anything-paddle; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.
```sh
pip install opencv-python pycocotools matplotlib
```


## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```python
from segment_anything_paddle import build_sam, SamPredictor
# if we load a pytorch model, we will autoconvert this to paddle model
predictor = SamPredictor(build_sam(checkpoint="</path/to/model.pth>"))
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```python
from segment_anything_paddle import build_sam, SamAutomaticMaskGenerator
# if we load a pytorch model, we will autoconvert this to paddle model
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="</path/to/model.pth>"))
masks = mask_generator.generate(<your_image>)
```

Additionally, masks can be generated for images from the command line:

```sh
python scripts/amg.py --checkpoint <path/to/sam/checkpoint> --input <image_or_folder> --output <output_directory>
```

See the examples notebooks on [using SAM with prompts](/notebooks/predictor_example.ipynb) and [automatically generating masks](/notebooks/automatic_mask_generator_example.ipynb) for more details.

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running 
```python
from segment_anything_paddle import sam_model_registry
# if we load a pytorch model, we will autoconvert this to paddle model
sam = sam_model_registry["<name>"](checkpoint="<path/to/checkpoint>")
```
Click the links below to download the checkpoint for the corresponding model name. The default model in bold can also be instantiated with `build_sam`, as in the examples in [Getting Started](#getting-started).

* **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
* `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## License
The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The Segment Anything project was made possible with the help of many contributors (alphabetical):

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry. 

```
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
