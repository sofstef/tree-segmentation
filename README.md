# Tree trunk segmentation with smartphones and neural networks

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>


This repository contains the code for my MRes project as part of the MRes+PhD in AI for Environmental risk at Cambridge University. 

In this project we employ a machine learning approach to improve the accuracy of automatic diameter estimates of forest trees using a mobile phone app that runs on a smartphone with a regular camera. We do this by collecting and curating a dataset that allows us to make use of the advantages of depth maps obtained from a LiDAR-equipped phone to refine depth maps obtained from a regular camera using a depth-from-motion algorithm developed by Google. We train a UNet model with a ResNet34 encoder for a binary image segmentation task of segmenting tree trunks from input depth data, and use the predicted segments to select depths to feed to an existing diameter estimation algorithm.


## Data 

The data collected for this project is freely available and can be found [here](https://doi.org/10.5281/zenodo.6787045). To reproduce the results obtained, download the data and place samples in `data/train/samples`, `data/train/segments`, `data/test/samples` and `data/test/segments`. Files with a `.tiff` extension are the target segments, and files without an extension are the inputs. There are also RGB images in the dataset with a `.jpeg` extension, but these are not used in training and testing. 

## Experiments

Once you've downloaded the data you can run any of the experiments set up in the configs, for example: 

```bash
$ python train.py --cfg=config/resnet34_experiment.yaml --o=my/output/dir
```


## Requirements
- Python 3.8+

## Getting started

Code implemented in Python 3.8.0

### Setting up environment

Create and activate environment

```
conda env create -f requirements/environment.yml -n env_name 
conda activate env_name 
(env_name)
```

## Project Organization
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements`
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_jqp_initial-data-exploration`.
│   ├── exploratory    <- Notebooks for initial exploration.
│   └── reports        <- Polished notebooks for presentations or intermediate results.
│
├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── sections       <- LaTeX sections. The report folder can be linked to your overleaf
|                         report with github submodules.
│
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── datasets       <- Scripts defining dataset classes
│   │
│   ├── datamodules    <- Scripts to set up and load data with PyTorch lightning datamodules
│   │
|   |
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── evaluation     <- Scripts defining functions used in evaluation
│
└── setup.cfg          <- setup configuration file for linting rules
```
---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
