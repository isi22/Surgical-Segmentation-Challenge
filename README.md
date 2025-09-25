# Surgical Tool Segmentation

This repository contains my solution for the machine vision challenge, focusing on the semantic segmentation of surgical instruments from the SAR-RARP50 dataset.

## Overview

The goal of this project is to accurately segment surgical tools, needles, and threads from video frames of robotic-assisted surgery. The solution implements a U-Net architecture with a pre-trained `EfficientNetB4` encoder using TensorFlow and Keras.

The project is structured into three main Jupyter Notebooks:

1. **Data Pre-processing:** A notebook to process the raw dataset into an efficient format for training.

2. **Model Training:** The main notebook that defines, trains, and saves the segmentation model.

3. **Model Inference:** A notebook to load the trained model and evaluate its performance on the test set.

## Setup Instructions



