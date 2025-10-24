# Surgical Tool Segmentation

<!--- Start of badges -->
<!-- Badges: python, keras, machinelearning, deeplearning -->

<!--- End of badges -->

<!--- Blurb
This project provides a machine vision solution for the semantic segmentation of surgical instruments from the SAR-RARP50 dataset. The objective is to generate pixel-level masks for robotic-assisted surgery videos, identifying prominent tools as well as small objects like needles and threads. The solution implements a U-Net architecture with a pre-trained EfficientNetB4 encoder using TensorFlow and Keras, achieving a final Test Set Mean IoU of 0.7439.
-->

<!--- Start of Thumbnail-->
<!--- src="Images/thumbnail.png" -->
<!--- End of Thumbnail-->

This repository contains my solution for the machine vision challenge, focusing on the semantic segmentation of surgical instruments from the SAR-RARP50 dataset.

## Overview

The goal of this project is to accurately segment surgical tools, needles, and threads from video frames of robotic-assisted surgery. The solution implements a U-Net architecture with a pre-trained `EfficientNetB4` encoder using TensorFlow and Keras.

The project is structured into three main Jupyter Notebooks:

1. **Data Pre-processing:** a notebook that converts the raw dataset into an optimised format for training by unzipping the original archives, extracting video frames that match the segmentation masks, and aggregating all images and masks into flat directories.
2. **Model Training:** the main notebook that defines, trains, and saves the segmentation model.
3. **Model Inference:** a notebook to load the trained model and evaluate its performance on the test set.

### **Results Summary**

 - Final Test Set Mean IoU: 0.7439

 - Validation Set Mean IoU: 0.7979

 - Training Set Mean IoU: 0.7552

## Running the Project in Google Colab

Training deep learning models for image segmentation is computationally expensive and requires a powerful GPU. Therefore, I used Google Colab for this projects, since it provides free access to high-performance hardware. 

### 1. Open the Notebooks in Colab

Click the badges below to open the training and inference notebooks directly in Google Colab.

 - [Data pre-processing notebook](https://colab.research.google.com/github/isi22/Surgical-Segmentation-Challenge/blob/main/notebooks/1_data_preprocessing.ipynb)
 - [Model training notebook](https://colab.research.google.com/github/isi22/Surgical-Segmentation-Challenge/blob/main/notebooks/2_model_training.ipynb)
 - [Model inference notebook](https://colab.research.google.com/github/isi22/Surgical-Segmentation-Challenge/blob/main/notebooks/3_model_inference.ipynb)

Ensure a GPU is enabled (Runtime -> Change runtime type -> Select a GPU -> Save) 

### 2. Run the Training Notebook
 - Open and run the `2_model_training.ipynb` notebook using the link above.
 - The "Environment Setup" and "Data Setup" cells will automatically clone this repository, install all dependencies, and download the pre-processed dataset (stored on Google Drive after pre-processing with `1_data_preprocessing`).
 - The notebook will then build the model and run the full training process.

### 3. Run the Inference Notebook
 - Open and run the `3_model_inference.ipynb` notebook using the link above.

 - The pre-processed training dataset is downloaded automatically, along with the training history and final trained version of the model (unet_efficientnetb4_best.keras) after 30 epochs.

 - The notebook will:
   -  display the training history (loss and mean IoU of the training and validation data throughout training)

   - evaluate the model's performance on the entire test set and display the final Test Loss and Test Mean IoU

   - show several visual examples of the model's predictions on test images

 ## Model and Training Strategy

 - **Architecture:** the model is a U-Net with a pre-trained EfficientNetB4 encoder, leveraging transfer learning for improved feature extraction.

 - **Loss Function:** a combined loss function of Dice Loss and Sparse Categorical Cross-entropy is used.

 - **Augmentation:** a data augmentation pipeline is implemented during training, which applies horizontal flips, shifts, rotations, gaussian noise, and brightness/contrast adjustments to improve the model's generalisation.

 - **Training:** the model was trained for 30 epochs. `ReduceLROnPlateau` was used to adjust the learning rate, which took effect after 10 epochs.


