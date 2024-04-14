# ECE324
Ryan Alizadeh
Jade Clement
Sergey Noritsyn

## Table of Contents

- [Introduction](#Introduction)
- [Requirments](#Requirments)
- [Usage](#usage)


## Introduction
This repository contains all the code and data generated for our ECE324 project investigating the use of synthetic data in augmenting medical data machine learning applications. Specifically, we attempt to train a DCGAN on a limited training set, then augment that dataset with images generated by the DCGAN. Finally, a classifier is trained on the augmented dataset, and we record the training history of that classifier as well as it's test accuracy.  Throughout the codebase, the following terminolgy is used:

- **CI ratio** This is the class imbalance ratio (ratio of positive data points / total data points) for the unaugmented dataset
- **NRS** or **Num real shots** This is the number of real data points in the unaugmented dataset
- **SR ratio** This is the ratio of synthetic to real data in the augmented dataset, or the amount of synthetic data that the DCGAN must generate

## Requirments

- Python 3.10 or higher
- numpy
- tensorflow
- sklearn
- keras
- medmnist (use `pip install medmnist`)
- matplotlib

## Usage

The main entrypoint to this code is `experiment.ipynb`, walking through that notebook will call all the relevant code to generate the key results.

1. First the `ConfigGenerator` class is used to generate a set of config files. For our research, we specify configs to cover a grid of ci ratios, NRS's, and sr ratios.
2. The config files are iterated over and loaded as an ExperimentConfig object. The DataLoader object uses the config to specify how to set up the limited dataset from the base MedMNIST dataset.
3. The ExperimentRunner class sets up the experiment for each config file and limited dataset by doing the following steps
    - determines how many synthetic images need to be generated
    - trains the DCGAN on the limited dataset
    - generates the required number of synthetic images
    - combines the synthetic images with the limited dataset to form the augmented dataset
    - trains the classifier on the augmented dataset
    - returns the training history of the classifier and the classifier accuracy
4. The history data is visualized using MatPlotLib

**IMPORTANT**
Running the training code could take a while (~1 hour)
