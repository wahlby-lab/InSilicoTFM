[![License](https://img.shields.io/github/license/wahlby-lab/insilicotfm?style=flat-square)](https://opensource.org/licenses/MIT)

<h1 align="center">In Silico Prediction of Cell Traction Forces</h1>
<h4 align="center">🧫 Prediction Cell Traction forces with Deep Learning 🤖</h4>
<h4 align="center">⭐️ The code will be uploaded any time soon! ⭐️</h4>

## Table of Contents

- [Introduction](#introduction)
- [Examples](#examples)
- [Quick Start Guide](#quick-start-guide)
- [Citation](#citation)

## Introduction
Traction Force Microscopy is a method to determine the tensions a biological cell conveys to the underlying surface. Typically, Traction Force Microscopy requires culturing cells on gels with fluorescent beads, followed by bead displacement calculations. We present a new method allowing to predict those forces from a regular fluorescent image of the cell. Using Deep Learning, we trained a Bayesian Neural Network adapted for pixel regression of the forces and show that it generalises on different cells of the same strain. The predicted forces are computed along with an approximated uncertainty, which shows whether the prediction is trustworthy or not. Using the proposed method could help estimating forces in the absence of non-trivial calculation of bead displacements and can also free one of the fluorescent channels of the microscope.

This repository gives you access to the code necessary to:
* Train a Bayesian Neural Network for Traction Force prediction.
* Test the neural network on other cell datasets.
* Generate the figures present in the published paper.

## Example
[![Prediction of Cell Traction Forces with uncertainty](http://img.youtube.com/vi/U9-Tn9ojXAU/0.jpg)](https://youtu.be/U9-Tn9ojXAU "In Silico Prediction of Cell Traction Forces - Uncertainty")

[![Prediction of Cell Traction Forces](http://img.youtube.com/vi/QhzNmrA42T4/0.jpg)](https://youtu.be/QhzNmrA42T4 "In Silico Prediction of Cell Traction Forces")

## Quick Start Guide

### Trying the traction force prediction out
We set up a Colab notebook for you to try predicting the traction forces on any images of your choice. The computation is done on the cloud for free, no need to install anything. Get started in 15 seconds.

You can access the notebook at the following address: https://colab.research.google.com/drive/16eX9UOFn4cQCL6hqcxgiX5QEm8yAhTaR

### Training the neural network
```bash
git clone https://github.com/wahlby-lab/InSilicoTFM.git
pipenv install
jupyter notebook
```

## Citation
Please cite the following paper if you are using or contributing to our method:

>Nicolas Pielawski, Jianjiang Hu, Staffan Strömblad, and Carolina Wählby. "In Silico Prediction of Cell Traction Forces" arXiv preprint arXiv:1910.xxxxx (2019).

```
@article{pielawski2019insilicotfm,
  title={In Silico Prediction of Cell Traction Forces},
  author={Pielawski, Nicolas and Hu, jianjiang and Str\"{o}mblad, Staffan and W\"{a}hlby, Carolina},
  journal={arXiv preprint arXiv:1910.xxxxx},
  year={2019}
}
```