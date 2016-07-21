# tf-dcgan

## Introduction

This repository contains a TensorFlow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434).

## Requirements

 - Python (2.7 or 3.5)
 - TensorFlow >= 0.8

## Usage

To train your own model, clone this repository and run:

```
python dcgan.py -t <train_dir> -o <output_dir>
```

This will snapshot weights as well as intermediate generations to the directory specified by `<output_dir>`. To generate samples from a trained model, simply omit the `-t` flag:

```
python dcgan.py -o <output_directory>
```

For more options, run:

```
python dcgan.py -h
```
