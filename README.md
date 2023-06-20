# PatchPerPix for Instance Segmentation

This repository is the official implementation of [PatchPerPix for Instance Segmentation](https://arxiv.org/abs/2001.07626).

 **Lisa Mais<sup>1</sup>, Peter Hirsch<sup>1</sup>, Dagmar Kainmueller**, ECCV2020</br>
<sup>1</sup>Authors contributed equally, listed in random order</br>

![PatchPerPix for Instance Segmentation](./README.assets/pipeline.png "PatchPerPix")

## Abstract
We present a novel method for proposal free instance segmentation that can handle sophisticated object shapes that span large parts of an image and form dense object clusters with crossovers. Our method is based on predicting dense local shape descriptors, which we assemble to form instances. All instances are assembled simultaneously in one go. To the best of our knowledge, our method is the first non-iterative method that yields instances that are composed of learnt shape patches. We evaluate our method on a diverse range of data domains, where it defines the new state of the art on four benchmarks, namely the ISBI 2012 EM segmentation benchmark, the BBBC010 C. elegans dataset, and 2d as well as 3d fluorescence microscopy datasets of cell nuclei. We show furthermore that our method also applies to 3d light microscopy data of drosophila neurons, which exhibit extreme cases of complex shape clusters.

## Installation

This package requires Python 3 and PyTorch.

**Note**
Previous versions (e.g., for the experiments published in our ECCV 2020 paper) require TensorFlow 1.x.
If you want to run older experiments please checkout the respective tag: [eccv2020](https://github.com/Kainmueller-Lab/PatchPerPix/tree/ea4e2d4)
If you have any questions, please open an issue (and mention that you're running the older code)

The recommended way is to install the package into your conda/python virtual environment. We recommend to use conda to install torch (tested with torch 1.13, but newer versions should work, too). The following instructions were tested on linux/ubuntu 20.04.


```
conda create --name ppp --yes
conda activate ppp
conda install python=3.9 pytorch-cuda torchvision torchaudio cudatoolkit -c pytorch -c nvidia --yes
git clone https://github.com/Kainmueller-Lab/PatchPerPix.git
cd PatchPerPix
PATH=/usr/local/cuda/bin:$PATH CUDA_ROOT=/usr/local/cuda pip install -e .
```

## Organization
- PatchPerPix: contains the code for our instance assembly pipeline to go from predictions to instances
- experiments: contains the training and prediction code to generate predictions and the main script; contains one sub-folder per application/dataset
  - `run_ppp.py`:
	- main script to run the experiments
	- command line arguments are used to select the experiment and the sub-task to be executed (training, inference etc, see below for an example)
	- the parameters for the network training and the postprocessing have to be defined in a config file ([example config file](https://github.com/Kainmueller-Lab/PatchPerPix/blob/master/experiments/flylight/setups/setup01/default.toml))
  - flylight: an example experiment for the FlyLight Instances Segmentation Benchmark Dataset
	- setups: here the different experiment setups are placed, the python scripts should not be called manually, but will be called by the main script
     - `train.py`: trains the network
     - `predict_no_gp.py`: prediction after training
     - `decode.py`: if ppp+dec is used, decode the predicted patch encodings to the full patches
     - `default.toml`: example configuration file
	 - `default_train_code.toml`: example configuration file that uses ppp+dec
	 - `torch_loss.py`: auxiliary file for the loss computation
	 - `torch_model.py`: auxiliary file for the torch model definition


## Data preparation

The code expects the data to be in the *zarr* format ([https://zarr.readthedocs.io/en/stable/](https://zarr.readthedocs.io/en/stable/)). It is similar to *hdf5*, but uses the underlying file system to enable parallel read and write). It expects all used arrays (e.g., raw image data and labels) to be placed in a single zarr file (organized into a hierarchy via groups, see [zarr documentation](https://zarr.readthedocs.io/en/stable/tutorial.html#groups)).
The names of the arrays have to be set in the config file (e.g., `raw_key` and `gt_key`) appropriately ([example zarr file](experiments/flylight/JRC_SS05008-20160318_24_B2_crop.zip)).
<!-- The dataset specific subfolders (e.g. [flylight](flylight)) contain further information on how to get and preprocess the data. -->


## Usage
The main script `run_ppp.py` (in the experiments folder) can be used to control all aspects of the experiments.

Example call:
```
python run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code.toml --do train validate_checkpoints predict decode label evaluate --app flylight --root ppp_experiments
```

With `--do TASK` you can set the sub-task that should be executed (or `all` for the whole pipeline), `--root PATH` sets the output directory, `--app APP` the experiment (e.g. flylight) and `--setup SETUP` the specific setup of that experiment (e.g. setup01).

The command above creates a time stamped experiment folder under the path specified by `--root`.
To continue training or for further validation or evaluation adapt the command. Change the `--config` parameter to point to the config file in the created experiment folder and remove the `--root` flag and replace it with the `-id` flag and point it to the created experiment folder. The tasks specified after `--do` depend on what you want to do:
```
python run_ppp.py --setup setup01 --config ppp_experiments/flylight_setup01_230614__123456/config.toml --do  validate_checkpoints predict decode label evaluate --app wormbodies -id experiments/flylight_setup01_230614__123456
```

### Available Sub-Tasks

| Task                   | Short Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------------|
| `all`                  | equal to `mknet train validate_checkpoints predict decode label postprocess evaluate`                  |
| `infer`                | equal to `predict decode label evaluate`                                                               |
| `mknet`                | creates a graph of the network (only for tensorflow 1)                                                 |
| `train`                | executes the training of the network                                                                   |
| `validate_checkpoints` | performs validation (over stored model checkpoints and a set of hyperparameters)                       |
| `validate`             | performs validation (for a specific model checkpoint and over a set of hyperparameters)                |
| `predict`              | executes the trained network in inference mode and computes predictions                                |
| `decode`               | decodes predicted patch encodings to full patches (only if model was trained to output encodings)      |
| `label`                | computes final instances based on predicted patches                                                    |
| `postprocess`          | post-processes predictions and predicted instances (optional, mostly for manual inspection of results) |
| `evaluate`             | compares predicted instances to ground truth instances and computes quantitative evaluation            |


## Results

(for more details on the results see [PatchPerPix for Instance Segmentation](https://arxiv.org/abs/2001.07626))

### BBBC010

([BBBC010: C. elegans live/dead assay](https://bbbc.broadinstitute.org/BBBC010))<br>
($S = \frac{TP}{TP+FP+FN}$; TP, FP, FN computed per image; averaged across images; localized using IoU)


| Method                   | avS<sub>[0.5:0.9:0.1]</sub> | S<sub>0.5</sub> | S<sub>0.6</sub> | S<sub>0.7</sub> | S<sub>0.8</sub> | S<sub>0.9</sub> |
|--------------------------|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Inst.Seg via Layering[1] | 0.754                       | 0.936           | 0.919           | 0.865           | 0.761           | 0.290           |
| PatchPerPix (ppp+dec)    | **0.816**                   | **0.960**       | **0.955**       | **0.931**       | **0.805**       | **0.428**       |

[1] results from: [Instance Segmentation of Dense and Overlapping Objects via Layering](https://arxiv.org/abs/2210.03551)


### ISBI2012

(server with leaderboard is down, but data is still available: [ISBI 2012 Segmentation Challenge](https://imagej.net/events/isbi-2012-segmentation-challenge)

| Method      | rRAND        | rINF         |
|-------------|--------------|--------------|
| PatchPerPix | **0.988290** | 0.991544     |
| MWS[2]      | 0.987922     | **0.991833** |
| MWS-Dense   | 0.979112     | 0.989625     |

[2] results from leaderboard (offline, see also [The Mutex Watershed: Efficient, Parameter-Free Image Partitioning](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.pdf))


### dsb2018

([Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/), train/val/test split defined by [Cell Detection with Star-convex Polygons](https://arxiv.org/abs/1806.03535))<br>
($S = \frac{TP}{TP+FP+FN}$; TP, FP, FN computed per image; averaged across images; localized using IoU)


| Method        | avS<sub>[0.5:0.9:0.1]</sub> | S<sub>0.1</sub> | S<sub>0.2</sub> | S<sub>0.3</sub> | S<sub>0.4</sub> | S<sub>0.5</sub> | S<sub>0.6</sub> | S<sub>0.7</sub> | S<sub>0.8</sub> | S<sub>0.9</sub> |
|---------------|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Mask R-CNN[3] | 0.594                       | -               | -               | -               | -               | 0.832           | 0.773           | 0.684           | 0.489           | 0.189           |
| StarDist[3]   | 0.584                       | -               | -               | -               | -               | 0.864           | 0.804           | 0.685           | 0.450           | 0.119           |
| PatchPerPix   | **0.693**                   | **0.919**       | **0.919**       | **0.915**       | **0.898**       | **0.868**       | **0.827**       | **0.755**       | **0.635**       | **0.379**       |

[3] results from [Cell Detection with Star-convex Polygons](https://arxiv.org/abs/1806.03535)


### nuclei3d

([https://doi.org/10.5281/zenodo.5942574](https://doi.org/10.5281/zenodo.5942574), train/val/test split defined by [Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy](https://arxiv.org/abs/1908.03636))<br>
($S = \frac{TP}{TP+FP+FN}$; TP, FP, FN computed per image; averaged across images; localized using IoU)


| Method         | avS<sub>[0.5:0.9:0.1]</sub> | S<sub>0.1</sub> | S<sub>0.2</sub> | S<sub>0.3</sub> | S<sub>0.4</sub> | S<sub>0.5</sub> | S<sub>0.6</sub> | S<sub>0.7</sub> | S<sub>0.8</sub> | S<sub>0.9</sub> |
|----------------|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| MALA[4]        | 0.381                       | 0.895           | 0.887           | 0.859           | 0.803           | 0.699           | 0.605           | 0.424           | 0.166           | 0.012           |
| StarDist3d[5]  | 0.406                       | 0.936           | 0.926           | 0.905           | **0.855**       | 0.765           | 0.647           | 0.460           | 0.154           | 0.004           |
| 3-label+cpv[6] | 0.425                       | **0.937**       | **0.930**       | **0.907**       | 0.848           | 0.750           | 0.641           | 0.473           | 0.224           | **0.035**       |
| PatchPerPix    | **0.436**                   | 0.926           | 0.918           | 0.900           | 0.853           | **0.766**       | **0.668**       | **0.493**       | **0.228**       | 0.027           |

[4] [Large Scale Image Segmentation with Structured Loss based Deep Learning for Connectome Reconstruction](https://arxiv.org/pdf/1709.02974.pdf), we computed the results<br>
[5] results from [Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy](https://arxiv.org/abs/1908.03636)<br>
[6] results from [An Auxiliary Task for Learning Nuclei Segmentation in 3D Microscopy Images](https://arxiv.org/abs/2002.02857)


### FlyLight

([The FlyLight Instance Segmentation Datset](https://kainmueller-lab.github.io/flylight_inst_seg_dataset/), train/val/test split defined by tba)

| Metrik         | short description              |
|----------------|--------------------------------|
| S              | average of avF1 and C          |
| avF1           | Multi-Threshold F1 Score       |
| C              | Average ground Truth coverage  |
| C<sub>TP</sub> | Average true positive coverage |
| FS             | Number of false splits         |
| FM             | Number of false merges         |

(for a precise definition see tba)

<br>Trained on *completely* labeled data, evaluated on *completely* labeled data and *partly* labeled data combined:
| Method                                                                                                | S | avF1 | C | C<sub>TP</TP> | FS | FM |
|-------------------------------------------------------------------------------------------------------|---|------|---|---------------|----|----|
| PatchPerPix&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |   |      |   |               |    |    |
|                                                                                                       |   |      |   |               |    |    |

Trained on *completely* labeled and *partly* labeled data combined, evaluated on *completely* labeled data and *partly* labeled data combined:
| Method               | S | avF1 | C | C<sub>TP</TP> | FS | FM |
|----------------------|---|------|---|---------------|----|----|
| PatchPerPix(+partly) |   |      |   |               |    |    |
|                      |   |      |   |               |    |    |


## Contributing

If you would like to contribute, have encountered any issues or have any suggestions, please open an issue on this GitHub repository.

All contributions  are welcome! The content in this repository is licensed under the MIT license.
