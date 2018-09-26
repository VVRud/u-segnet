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
