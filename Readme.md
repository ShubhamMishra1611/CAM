# Image Classification using Grad-CAM and ResNet50

This is a Python script that demonstrates how to use the Grad-CAM algorithm to visualize the activation of a convolutional neural network (CNN) for image classification. It uses the ResNet50 pre-trained model provided by Keras as the CNN, and shows how to extract the feature maps and the class activation maps (CAMs) of an input image.

## Requirements

This script requires the following packages to be installed:

- TensorFlow 2.x
- Keras 2.x
- NumPy
- Matplotlib
- SciPy

## Usage

1. Clone or download the repository.
2. Open the script in a Python environment, e.g. Jupyter Notebook.
3. Run the script.

The script will randomly select an image from the `images` folder, use the ResNet50 model to classify the image, and then generate the CAM for the predicted class using Grad-CAM.

## Acknowledgments

This script is based on the following resources:

- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [Keras Pre-trained Models](https://keras.io/api/applications/)

