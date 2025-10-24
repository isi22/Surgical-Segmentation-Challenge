# Surgical Tool Segmentation

<!--- Start of badges -->
<!-- Badges: python, keras, machinelearning, deeplearning -->

<p align="left">
<img alt="Deeplearning" src="https://img.shields.io/badge/-Deep_Learning-333333.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBMaWNlbnNlOiBNSVQuIE1hZGUgYnkgRXNyaTogaHR0cHM6Ly9naXRodWIuY29tL0VzcmkvY2FsY2l0ZS11aS1pY29ucyAtLT4KPHN2ZyB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBmaWxsPSIjZmZmZmZmIiBkPSJNMjAuNSA5YTMuNDkgMy40OSAwIDAgMC0zLjQ1IDNoLTEuMWEyLjQ5IDIuNDkgMCAwIDAtNC4zOTYtMS4wNTJMOC44NzggOS43MzFsMy4xNDMtNC4yMjVhMi40NTggMi40NTggMCAwIDAgMi45OC0uMDE5TDE3LjMzOSA4SDE2djFoM1Y2aC0xdjEuMjQzbC0yLjMzNi0yLjUxMkEyLjQ3MyAyLjQ3MyAwIDAgMCAxNiAzLjVhMi41IDIuNSAwIDAgMC01IDAgMi40NzQgMi40NzQgMCAwIDAgLjM0MyAxLjI0M0w3Ljk0NyA5LjMwOCA0Ljk1NSA3Ljk0N2EyLjQwNCAyLjQwNCAwIDAgMC0uMTYxLTEuNDM4bDMuNzA0LTEuMzg1LS40NCAxLjM3MS45NDIuMzMzTDEwIDQgNy4xNzIgM2wtLjMzNC45NDMgMS4wMS4zNTctMy42NTkgMS4zNjhhMi40OTggMi40OTggMCAxIDAtLjY4MiA0LjExN2wyLjA4NSAyLjY4OC0yLjA1MyAyLjc2YTIuNSAyLjUgMCAxIDAgLjg3IDMuODY0bDMuNDg0IDEuNTg3LTEuMDU1LjM3My4zMzQuOTQzTDEwIDIxbC0xLTIuODI4LS45NDMuMzMzLjQzNSAxLjM1NC0zLjYwOC0xLjY0NUEyLjQ3MSAyLjQ3MSAwIDAgMCA1IDE3LjVhMi41IDIuNSAwIDAgMC0uMDU4LS41MjdsMy4wNTMtMS40MDUgMy40NzYgNC40OGEyLjQ5OCAyLjQ5OCAwIDEgMCA0LjExMy4wNzVMMTggMTcuNzA3VjE5aDF2LTNoLTN2MWgxLjI5M2wtMi40MTYgMi40MTZhMi40NjYgMi40NjYgMCAwIDAtMi42NjctLjA0N2wtMy4yODMtNC4yMyAyLjU1NC0xLjE3NkEyLjQ5NCAyLjQ5NCAwIDAgMCAxNS45NSAxM2gxLjFhMy40OTMgMy40OTMgMCAxIDAgMy40NS00em0tNy03QTEuNSAxLjUgMCAxIDEgMTIgMy41IDEuNTAyIDEuNTAyIDAgMCAxIDEzLjUgMnptMCAxOGExLjUgMS41IDAgMSAxLTEuNSAxLjUgMS41MDIgMS41MDIgMCAwIDEgMS41LTEuNXpNMSA3LjVhMS41IDEuNSAwIDEgMSAyLjQ1NyAxLjE0NWwtLjE0NC4xMTJBMS40OTYgMS40OTYgMCAwIDEgMSA3LjV6bTMuMzIgMS43MDNhMi41MDcgMi41MDcgMCAwIDAgLjI2NC0uMzI2bDIuNzUyIDEuMjUxLTEuMTI0IDEuNTEyek0yLjUgMTlBMS41IDEuNSAwIDEgMSA0IDE3LjUgMS41MDIgMS41MDIgMCAwIDEgMi41IDE5em0yLjAzNy0yLjk0MWEyLjUxOCAyLjUxOCAwIDAgMC0uMTkzLS4yMzRsMS44ODUtMi41MzIgMS4xMzYgMS40NjR6bTMuNzYtMS43MzFMNi44NDkgMTIuNDZsMS40Mi0xLjkwOEwxMS4xIDExLjg0YTIuMjkgMi4yOSAwIDAgMC0uMDMzIDEuMjEzek0xMy41IDE0YTEuNSAxLjUgMCAxIDEgMS41LTEuNSAxLjUwMiAxLjUwMiAwIDAgMS0xLjUgMS41em03IDFhMi41IDIuNSAwIDEgMSAyLjUtMi41IDIuNTAyIDIuNTAyIDAgMCAxLTIuNSAyLjV6bTEuNS0yLjVhMS41IDEuNSAwIDEgMS0xLjUtMS41IDEuNTAyIDEuNTAyIDAgMCAxIDEuNSAxLjV6Ii8+PHBhdGggZmlsbD0ibm9uZSIgZD0iTTAgMGgyNHYyNEgweiIvPjwvc3ZnPg==&style=flat-square" />
 <img alt="Keras" src="https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white&style=flat-square" />
 <img alt="Machinelearning" src="https://img.shields.io/badge/-Machine_Learning-333333.svg?logo=data:image/svg+xml;base64,PCEtLSBMaWNlbnNlOiBBcGFjaGUuIE1hZGUgYnkgQ2FyYm9uIERlc2lnbjogaHR0cHM6Ly9naXRodWIuY29tL2NhcmJvbi1kZXNpZ24tc3lzdGVtL2NhcmJvbiAtLT4KPHN2ZyB3aWR0aD0iMzJweCIgaGVpZ2h0PSIzMnB4IiB2aWV3Qm94PSIwIDAgMzIgMzIiIGlkPSJpY29uIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuY2xzLTEgewogICAgICAgIGZpbGw6IG5vbmU7CiAgICAgIH0KICAgIDwvc3R5bGU+CiAgPC9kZWZzPgogIDxwYXRoIGZpbGw9IiNmZmZmZmYiIGQ9Ik0yNywyNGEyLjk2MDksMi45NjA5LDAsMCwwLTEuMjg1NC4zMDA4TDIxLjQxNDEsMjBIMTh2MmgyLjU4NTlsMy43MTQ2LDMuNzE0OEEyLjk2NjUsMi45NjY1LDAsMCwwLDI0LDI3YTMsMywwLDEsMCwzLTNabTAsNGExLDEsMCwxLDEsMS0xQTEuMDAwOSwxLjAwMDksMCwwLDEsMjcsMjhaIi8+CiAgPHBhdGggZmlsbD0iI2ZmZmZmZiIgZD0iTTI3LDEzYTIuOTk0OCwyLjk5NDgsMCwwLDAtMi44MTU3LDJIMTh2Mmg2LjE4NDNBMi45OTQ3LDIuOTk0NywwLDEsMCwyNywxM1ptMCw0YTEsMSwwLDEsMSwxLTFBMS4wMDA5LDEuMDAwOSwwLDAsMSwyNywxN1oiLz4KICA8cGF0aCBmaWxsPSIjZmZmZmZmIiBkPSJNMjcsMmEzLjAwMzMsMy4wMDMzLDAsMCwwLTMsMywyLjk2NTcsMi45NjU3LDAsMCwwLC4zNDgxLDEuMzczTDIwLjU5NTcsMTBIMTh2MmgzLjQwNDNsNC4zOTg5LTQuMjUyNEEyLjk5ODcsMi45OTg3LDAsMSwwLDI3LDJabTAsNGExLDEsMCwxLDEsMS0xQTEuMDAwOSwxLjAwMDksMCwwLDEsMjcsNloiLz4KICA8cGF0aCBmaWxsPSIjZmZmZmZmIiAgZD0iTTE4LDZoMlY0SDE4YTMuOTc1NiwzLjk3NTYsMCwwLDAtMywxLjM4MjNBMy45NzU2LDMuOTc1NiwwLDAsMCwxMiw0SDExYTkuMDEsOS4wMSwwLDAsMC05LDl2NmE5LjAxLDkuMDEsMCwwLDAsOSw5aDFhMy45NzU2LDMuOTc1NiwwLDAsMCwzLTEuMzgyM0EzLjk3NTYsMy45NzU2LDAsMCwwLDE4LDI4aDJWMjZIMThhMi4wMDIzLDIuMDAyMywwLDAsMS0yLTJWOEEyLjAwMjMsMi4wMDIzLDAsMCwxLDE4LDZaTTEyLDI2SDExYTcuMDA0Nyw3LjAwNDcsMCwwLDEtNi45Mi02SDZWMThINFYxNEg3YTMuMDAzMywzLjAwMzMsMCwwLDAsMy0zVjlIOHYyYTEuMDAwOSwxLjAwMDksMCwwLDEtMSwxSDQuMDhBNy4wMDQ3LDcuMDA0NywwLDAsMSwxMSw2aDFhMi4wMDIzLDIuMDAyMywwLDAsMSwyLDJ2NEgxMnYyaDJ2NEgxMmEzLjAwMzMsMy4wMDMzLDAsMCwwLTMsM3YyaDJWMjFhMS4wMDA5LDEuMDAwOSwwLDAsMSwxLTFoMnY0QTIuMDAyMywyLjAwMjMsMCwwLDEsMTIsMjZaIi8+CiAgPHJlY3QgaWQ9Il9UcmFuc3BhcmVudF9SZWN0YW5nbGVfIiBkYXRhLW5hbWU9IiZsdDtUcmFuc3BhcmVudCBSZWN0YW5nbGUmZ3Q7IiBjbGFzcz0iY2xzLTEiIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIvPgo8L3N2Zz4K&style=flat-square" />
 <img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square" />
</p>
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


