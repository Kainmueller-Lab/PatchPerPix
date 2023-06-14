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

## Organization
- PatchPerPix: contains the main code to go from predictions to instances
- experiments: contains the training and prediction code to generate predictions
  - `run_ppp.py`:
	- master script to run the experiments
	- command line arguments are used to select the experiment and the sub-task to be executed (training, inference etc, see below for an example)
	- the parameters for the network training and the postprocessing are supplied in a config file
  - flylight: an example experiment for the FlyLight Instances Segmentation Benchmark Dataset
	- setups: here the different experiment setups are placed, the python scripts should not be called manually, but will be called by the master script
     - `train.py`: trains the network
     - `predict_no_gp.py`: inference after training
     - `decode.py`: if ppp+dec is used, decode the predicted foreground codes to the full patches
     - `default.toml`: example configuration file
	 - `default_train_code.toml`: example configuration file that uses ppp+dec
	 - `torch_loss.py`: auxiliary file for the loss computation
	 - `torch_model.py`: auxiliary file for the torch model definition


## Data preperation

The code expects the data to be in the *zarr* format ([https://zarr.readthedocs.io/en/stable/](https://zarr.readthedocs.io/en/stable/) similar to *hdf5*, but uses the underlying file system to enable parallel read and write)
<!-- The dataset specific subfolders (e.g. [flylight](flylight)) contain further information on how to get and preprocess the data. -->


## Usage
The master script `run_ppp.py` (in the experiments folder) can be used to control all aspects of the experiments.

Example call:
```
python run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code.toml --do train validate_checkpoints predict decode label evaluate --app flylight --root ppp_experiments
```

With `--do TASK` you can set the sub task that should be executed (or `all` for the whole pipeline), `--root PATH` sets the output directory, `--app APP` the experiment (e.g. flylight) and `--setup SETUP` the specific setup of that experiment (e.g. setup01).

The command above creates a time stamped experiment folder under the path specified by `--root`.
To continue training or for further validation or evaluation adapt the command. Change the `--config` parameter to point to the config file in the experiment folder and remove the `--root` flag and replace it with the `-id` flag and point it to the experiment folder. The tasks specified after `--do` depend on what you want to do:
```
python run_ppp.py --setup setup01 --config ppp_experiments/flylight_setup01_230614__123456/config.toml --do  validate_checkpoints predict decode label evaluate --app wormbodies -id experiments/flylight_setup01_230614__123456
```


## Results

*coming soon*

## Contributing

If you would like to contribute, have encountered any issues or have any suggestions, please open an issue on this GitHub repository.

All contributions  are welcome! The content in this repository is licensed under the MIT license.
