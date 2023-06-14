# PatchPerPix for Instance Segmentation

This repository is the official implementation of [PatchPerPix for Instance Segmentation](https://arxiv.org/abs/2001.07626).

 **Lisa Mais<sup>1</sup>, Peter Hirsch<sup>1</sup>, Dagmar Kainmueller**, ECCV2020</br>
<sup>1</sup>Authors contributed equally, listed in random order</br>

![PatchPerPix for Instance Segmentation](./README.assets/pipeline.png "PatchPerPix")

## Abstract
We present a novel method for proposal free instance segmentation that can handle sophisticated object shapes that span large parts of an image and form dense object clusters with crossovers. Our method is based on predicting dense local shape descriptors, which we assemble to form instances. All instances are assembled simultaneously in one go. To our knowledge, our method is the first non-iterative method that yields instances that are composed of learnt shape patches. We evaluate our method on a diverse range of data domains, where it defines the new state of the art on four benchmarks, namely the ISBI 2012 EM segmentation benchmark, the BBBC010 C. elegans dataset, and 2d as well as 3d fluorescence microscopy datasets of cell nuclei. We show furthermore that our method also applies to 3d light microscopy data of drosophila neurons, which exhibit extreme cases of complex shape clusters.

## Installation

This package requires Python 3 and PyTorch.

**Note**
Previous versions (e.g., for the experiments published in our ECCV 2020 paper) require TensorFlow 1.x.
If you want to run older experiments please checkout the respective tag: [eccv2020](https://github.com/Kainmueller-Lab/PatchPerPix/tree/ea4e2d4)
If you have any questions, please open an issue (and mention that you're running the older code)

The recommended way is to install the package into your conda/python virtual environment. We recommend to use conda to install torch.

```
conda create --name ppp
conda activate ppp
conda install python=3.9 pytorch-cuda torchvision torchaudio cudatoolkit -c pytorch -c nvidia
git clone https://github.com/Kainmueller-Lab/PatchPerPix.git
cd PatchPerPix
pip install -e .
```


## Training


## Evaluation


## Results

## Contributing

If you would like to contribute, have encountered any issues or have any suggestions, please open an issue on this GitHub repository.

All contributions  are welcome! The content in this repository is licensed under the MIT license.
