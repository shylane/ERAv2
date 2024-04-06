# Convolutional Neural Network for MNIST Dataset

This repository contains the code for training a Convolutional Neural Network (CNN) that classifies handwritten digits from on the MNIST dataset.  
The code is organized into separate files to promote modularity and reusability.

## Files

- `model.py`: This file contains the neural network architecture.
- `utils.py`: This file contains utility functions for data loading, training and evaluation.
- `S5.ipynb`: This Jupyter notebook contains the code for training and testing the CNN.
- `README.md`: This file (provides instructions and overview).

## Requirements

- Python 3.x
- PyTorch 1.x
- torchvision
- matplotlib
- tqdm

## Usage

1. Clone this repository.
2. Install required libraries: `pip install torch torchvision matplotlib tqdm`
3. Open the Jupyter Notebook `S5.ipynb` and follow the instructions within the notebook to train and test the CNN, and see the summary.

## Key Features

* Data augmentation for training data.
* Clear separation of model architecture and training logic.
* Visual tracking of training and validation accuracy and loss.
* Optional model summary using `torchsummary`.

## License

This project is licensed under the MIT License.
