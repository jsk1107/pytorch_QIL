# Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss(QIL)

This is the unofficial implementation of [Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss](https://arxiv.org/abs/1808.05779)

describe more

## Results(ImageNet-1K)
|   *model*   | *w_bit* | *a_bit* | Accuracy(%) |
|:-----------:|:-------:|:-------:|:-----------:|
| `resnet-18` |   32    |   32    |    70.21    |
| `resnet-18` |    5    |    5    |    70.25    |
| `resnet-18` |    4    |    4    |    70.22    |
| `resnet-18` |    3    |    3    |    69.10    |
| `resnet-18` |    2    |    2    |    64.94    |
- ResNet is Pre-activation Resnet

## Getting started

### install

```shell
git clone https://github.com/jsk1107/pytorch_QIL.git
cd pytorch_QIL
pip install -r requirement.txt
```

The training environment consists of the following.
- Ubuntu 18.04
- A100 GPU
  - We use TensorCore(TF32) supported by the A100(Ampere architecture).
- Python 3.7
- Pytorch 1.8

### Data preparation

We use ILSVRC2012 dataset. After download, prepare the data structure by separating it into train and validation.
See this site [Preparation of ImageNet (ILSVRC2012)](https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a)

### Train

According to the paper, it is mentioned that it performs progressive learning(32 -> 5 -> 4 -> 3 -> 2).
So, If you want to obtain 2bit weight, Training must be carried out for all bits from 32bit to 3bit.

