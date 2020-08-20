# PatchPerPix for Instance Segmentation

PDF: [PatchPerPix for Instance Segmentation](https://arxiv.org/abs/2001.07626)


**Lisa Mais<sup>1</sup>, Peter Hirsch<sup>1</sup>, Dagmar Kainmueller**, ECCV2020</br>
<sup>1</sup>Authors contributed equally, listed in random order</br>

![PatchPerPix for Instance Segmentation](./README.assets/pipeline.png "PatchPerPix")

## Abstract
We present a novel method for proposal free instance segmentation that can handle sophisticated object shapes that span large parts of an image and form dense object clusters with crossovers. Our method is based on predicting dense local shape descriptors, which we assemble to form instances. All instances are assembled simultaneously in one go. To our knowledge, our method is the first non-iterative method that yields instances that are composed of learnt shape patches. We evaluate our method on a diverse range of data domains, where it defines the new state of the art on four benchmarks, namely the ISBI 2012 EM segmentation benchmark, the BBBC010 C. elegans dataset, and 2d as well as 3d fluorescence microscopy datasets of cell nuclei. We show furthermore that our method also applies to 3d light microscopy data of drosophila neurons, which exhibit extreme cases of complex shape clusters.


## Disclaimer

The readme will be extended in the near future.
In the meanwhile, if you have questions regarding the code, on how to run it or run into problems, please open an issue!


## Installation

This package requires Python 3 and TensorFlow 1.x (we do not support Tensorflow 2.x). It contains the core code for the neural network model and our instance segmentation method.

### Core
The recommended way is to install the package into your conda/python virtual environment.

```
conda activate <<your-env-name>>
git clone https://github.com/Kainmueller-Lab/PatchPerPix.git
cd PatchPerPix
pip install -e .
```

### Experiments
Example scripts to train a network and run the method are in a separate repository:
[PatchPerPix_experiments](https://github.com/Kainmueller-Lab/PatchPerPix_experiments).</br>
Please continue there after finishing the installation.
