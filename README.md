# Monkeypox Measles and Chickenpox Detection using Deep Learning

## Overview
This project focuses on classifying images into monkeypox, chickenpox, measles and normal  category through a deep learning model, specifically a fine-tuned ResNet model, and deploying it with Streamlit for easy accessibility. The project aims to provide a quick and reliable method for disease detection using image classification. Various models like DenseNet , Xception , VGG16, VGG19 were tested and fine tuned before finalizing ResNet which gave the best results.

## Features
•	Deep Learning Model: Utilizes a fine-tuned ResNet model for high accuracy in detecting monkeypox.
•	Dataset: Includes images of chickenpox, measles, monkeypox, and normal skin conditions.
•	Data Augmentation: Enhances the dataset through techniques like rotation, zooming, and flipping.
•	Streamlit Deployment: User-friendly web interface for uploading images and getting instant predictions.

## Usage
Users can upload an image through the Streamlit app to classify whether the skin condition is monkeypox, chickenpox, measles, or normal. The model will display the prediction along with the probabilities for each class.

## Note :
Since the Streamlit App cannot handle a lot of traffic as of now , feel free to load the local_collab_run.ipynb in your local environment where you can use the provided .h5 file and test on various images.
