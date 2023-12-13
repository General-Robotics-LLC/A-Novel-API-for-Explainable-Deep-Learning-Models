# GNN for Molecular Structures: README
This README provides guidance on using the provided Python code, which includes a Graph Neural Network (GNN) for analyzing molecular structures. The code features various neural network components and training procedures tailored for molecular data.
Also, four jupyter files include the result in the paper (different performances with different query nodes)
## Overview
The code includes:

Loading and preprocessing molecular structure data.
Defining Feed-Forward Neural Network (FNN) layers.
Creating a multi-head attention mechanism for GNNs.
Implementing a GNN model (GNN_vanilla) for processing molecular data.
Training and evaluation loops for the GNN model.
Activation and gradient hook functions for model introspection.
Dependencies
Ensure you have the following Python libraries installed:

numpy
random
pandas
torch
torch_geometric
sklearn
Key Components
Data Loading
The code starts by loading molecular structure data. Replace load_dataset_by_name with your dataset loading function:

dataset_name = 'DILI'
data = load_dataset_by_name(dataset_name)
# Neural Network Definitions
## FNN Layers
Two types of FNN layers are defined: FNN_sequential and FNN. These layers can be used to construct neural networks with different depths and configurations.

## Multihead Attention
The multihead class is a custom implementation of the multi-head attention mechanism, crucial for capturing complex relationships in molecular structures.

## GNN Model
GNN_vanilla is the main GNN model class. It integrates the previously defined neural network layers and attention mechanisms to process molecular graphs.

## Training and Evaluation
The code includes a training loop where the model is trained on a dataset, and the validation loss is monitored. Hook functions are used to capture activations and gradients for analysis.

## Hyperparameter Tuning
A combination of different hyperparameters is tested in the training loop. Adjust these parameters according to your dataset and computational resources.

## Execution Steps
Load your molecular dataset.
Define the GNN model and necessary components.
Run the training and evaluation loop, monitoring performance metrics like loss and mean squared error.
Optionally, analyze the activations and gradients captured by the hook functions.
## Customization
Modify the neural network architecture and hyperparameters based on your specific dataset.
Replace dataset loading and splitting functions as per your data processing pipeline.
By following these guidelines, you should be able to utilize the provided code effectively for molecular structure analysis using Graph Neural Networks.
