# AnimeNet
**This is a final project for NYU deep learning course**
The crawler code can be found at https://github.com/litousteven/pixiv_crawler/tree/master

## Code Introduction
There are mainly 4 parts of our Code and they also includes most of changes we did to the original Real-ESRGAN project:
1. anime_net/data I implement DegradationSimple, which is new degradation method I used on anime-style images.
2. anime_net/archs I implement Ovtave Block and the MixNet. MixNet uses both RDDB blocks and Ovtave Blocks to build a mix net.
3. anime_net/models I implement AnimeNet training code, which can adjust the loss function more flexible with config file. 
Then I can train my model by 3 stages in easy way.
4. scripts/generate_pair I write that script to generate validation pairs, 
which will be used to test the model during the training and the output images are used to show our training process.

## Environment Prepare

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

```bash
# Install basicsr - https://github.com/xinntao/BasicSR
pip install basicsr
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
```

(replace pip install with conda install if you are using  [Anaconda](https://www.anaconda.com/download/#linux))

To use GPU, please install Pytorch GPU version.

## Predict

```bash
python inference_anime.py [--input inputs] [--output results]
```

The inference will use AnimeNet model by default and use "inputs" folder as input and "results" folder as output.

## Train your model

### Build Dataset

Here are steps for data preparation.

#### Step 1: [Optional] Generate multi-scale images

You can use the [scripts/generate_multiscale_DF2K.py](scripts/generate_multiscale_DF2K.py) script to generate multi-scale images. <br>
Note that this step can be omitted if you just want to have a fast try. (I do not use this step on my Pixiv dataset)

```bash
python scripts/generate_multiscale_DF2K.py --input datasets/DF2K/DF2K_HR --output datasets/DF2K/DF2K_multiscale
```

#### Step 2: [Optional] Crop to sub-images

This step is optional if your IO is enough or your disk space is limited.

You can use the [scripts/extract_subimages.py](scripts/extract_subimages.py) script. Here is the example:

```bash
python scripts/extract_subimages.py --input datasets/DF2K/DF2K_multiscale --output datasets/DF2K/DF2K_multiscale_sub --crop_size 400 --step 200
```

#### Step 3: Prepare a txt for meta information

You need to prepare a txt file containing the image paths. The following are some examples in `meta_info_DF2Kmultiscale+OST_sub.txt` (As different users may have different sub-images partitions, this file is not suitable for your purpose and you need to prepare your own txt file):

```txt
DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...
```

You can use the [scripts/generate_meta_info.py](scripts/generate_meta_info.py) script to generate the txt file. <br>
You can merge several folders into one meta_info txt. Here is the example:

```bash
python scripts/generate_meta_info.py --input datasets/DF2K/DF2K_HR, datasets/DF2K/DF2K_multiscale --root datasets/DF2K, datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```



#### Step 4: Generate Validation Pair 

For only training data, the above 3 steps are enough, while for validation data we need 1 more steps.

We need to degradation data and provide several pairs of ground-truth image and low-quality image.

First config valid_animenet_x4plus.yml just like training config but replace the training dataset meta_info path with the validation dataset meta_info path.

Then run the CMD:

```bash
python scripts generate_pair.py -opt ../options/valid_animenet_x4plus.yml
```

Two folders of images will be generated: "valid_pair_gt" with ground-truth images and "valid_pair_lq" with low-quality images.



### Train

run CMD:

```
python realesrgan/train.py -opt options/train_animenet_x4plus.yml --auto_resume
```
To train our model stage by stage, we only need to do some adjustment to the config file (train_animenet_x4plus.yml). 
When we comment the discriminator config, the loss won't be added to the final loss.


