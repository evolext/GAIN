<h2 align='center'>Generative Adversarial Imputation Networks (GAIN) Pytorch Implementation</h2>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#related">Related</a> •
  <a href="#features">Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#data-used">Data used</a>
</p>

## Description

The project is a PyTorch implementation of [Generative Adversarial Imputation Networks](https://arxiv.org/pdf/1806.02920.pdf) algorithm for imputation of missing values in data.

## Related

The core of the project is another [PyTorch implementation](https://github.com/dhanajitb/GAIN-Pytorch/) that has been refactored and upgraded (see below).

## Features

Refactoring fixes (in order of importance):

- Fixed bug of training by batches in the train() method
- All model elements have been compiled into a separate class in the .py module to allow import into other projects
- Fixed Hint matrix calculation according to the original article
- Generator and Discriminator are defined using inheritance from the torch.nn.module class
- Added option to use the model after training (evaluation() method)
- Added the option to use EarlyStopping in the training process
- Added the ability to change the device during operation
- Saving the history of error changes in the training process
- Setting the seed value to reproduce the results

## How To Use

Python 3.9 was used during development, other dependencies are listed in the requirements.txt file at the root of the project.<br>
An example of using the GAIN class for data imputation is presented in the usage_example.py module.

## Data used

The following data were used in the original article and in the development process:

- [UCI Letter](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
- [UCI Spam](https://archive.ics.uci.edu/ml/datasets/Spambase)











