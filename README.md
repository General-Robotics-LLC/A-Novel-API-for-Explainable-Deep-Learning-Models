# M.I.N.E.R.V.A.

Multifaceted
Integrated
Networked
Educational
Resource and
Virtual
Assistant

## Overview
This README details the use of the Python class `ExplainableNode`, designed as a part of our AI research project. This class is a separate Python illustration of the larger C++ class, focusing on enhancing model interpretability.

### ExplainableNode.py

#### Overview
`ExplainableNode` is a Python class designed for enhancing the interpretability of neural network models, specifically implemented for use with PyTorch. It provides functionalities for tracking and analyzing the activations and gradients of specified layers within a model during training, thereby aiding in understanding the model's decision-making process.

#### Features
1. **Initialization**: The class is initialized with options to compute two metrics - Per-Sample Magnitude (PSM) and Layer-wise Conditional Covariance (LCC).

2. **Hooks Registration**: 
    - `register_hooks`: Registers forward and backward hooks to specified layers in a model. These hooks capture the outputs and gradients for each layer during the forward and backward passes.

3. **Hook Functions**: 
    - `forward_hook_fn`: Captures the output of layers during the forward pass.
    - `backward_hook_fn`: Captures the gradients of layers during the backward pass.

4. **Metrics Calculation**:
    - `calculate_metrics`: Calculates the specified interpretability metrics (PSM and LCC) for each layer and appends them to the epoch metrics.
    - `get_aggregated_epoch_metrics`: Aggregates the metrics over epochs, providing a mean value for each metric across all epochs.

5. **Utility Functions**: 
    - `_compute_psm`: Internal function to compute the Per-Sample Magnitude for each layer.
    - `_compute_lcc`: Internal function to compute the Layer-wise Conditional Covariance.
    - `clear_values`: Clears stored activations and gradients.
    - `remove_hooks`: Removes all registered hooks from the model.

#### Usage
The `ExplainableNode` class is used by instantiating it with a model and the layers of interest. It then attaches hooks to these layers to monitor their outputs and gradients during training. After training, the class can compute and provide interpretability metrics that help in understanding how different layers of the model are behaving and contributing to the final output.

`run.py` contains the step needed for inilization and replicate the experiment using Expplainable Node on ResNet.
`train.py` contains the adapted version of train function that allows for the application of Explainable Node.
