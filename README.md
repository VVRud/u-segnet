# U-SegNet
[U-SegNet](https://arxiv.org/abs/1806.04429) is a segmentation network proposed by the research group of the Indraprastha Institute of Information Technology-Delhi. It is based on [U-Net](https://arxiv.org/abs/1505.04597) and [SegNet](https://arxiv.org/abs/1511.00561) architectures.

From **U-Net**:

- skip connection on the first layer.

From **SegNet**:

- downsamping using pooling;
- upsampling using unpooling with indices from pool operation.

---

## Architecture

Network consists of 2 blocks:

1. Block for **downsampling** consists of: N times (in this implenentation - 2) of block [Convolution -> Batch Normalization -> RELU] and Pooling operation.
2. Block for **upsampling** consists of: an UnPooling operation and N times (in this implenentation - 2) of block [Convolution -> Batch Normalization -> RELU].

To do unpooling you need to save indices from pooling operation.

Next image illustrates architecture proposed in paper.

<div align="center">
    <img align="center" width="800" src="https://github.com/vvrud/DRU-DL-Project-Structure/blob/master/figures/arch.png?raw=true">
</div>

---

## Differences from paper

In paper from IIIT-Delhi you can see that they use sliding window to predict every pixel's class, but I've implemented it for one single image.

---

# Usage

1. Put train and test data into data/u_segnet folder.<br/>
Path to train images and masks should be as follows:<br/>
``data/u_segnet/(train, test)/(images, masks)``
2. Via command line run commands below. It will take some time to create folders for train/dev/test splits with resized images in them.<br/>

```bash
cd home/$user/**path/to/project**/data/u_segnet/
python prepare_u_segnet.py --config ../../configs/u_segnet.json
```

3. Next commands will change your working dirrectory and start training cycle.

```bash
cd home/$user/**path/to/project**/mains/
python u_segnet_runner.py --config ../configs/u_segnet.json
```
