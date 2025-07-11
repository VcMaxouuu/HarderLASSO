# HarderLASSO

HarderLASSO is an advanced feature selection framework that combines neural networks with LASSO-style regularization. 
It provides a flexible approach to feature selection across various machine learning tasks, including regression, classification, and survival analysis.

## Features
- **Feature Selection**: Automatically identifies the most relevant features in your dataset
- **Task Flexibility**: Supports regression, classification, and survival analysis
- **Model Flexibility**: Can be configured as linear models or neural networks with multiple hidden layers
- **Automatic Tuning**: Automatically determines the optimal regularization parameter
- **Easy to Use**: Simple scikit-learn-like API for model fitting and evaluation
- **Interpretability**: Provides clear identification of selected features

## Quick Start
A quick start guide can be found under the `exemples.ipynb` notebook.

## Available Models
- `HarderLASSORegressor`: For regression tasks
- `HarderLASSOClassifier`: For classification tasks
- `HarderLASSOCox`: For survival analysis

## Parameters

All models share these common parameters:

- `hidden_dims`: Tuple of integers specifying the number of neurons in each hidden layer. Use `None` for linear models.
- `lambda_qut`: Regularization parameter. If `None`, it's determined automatically.
- `penalty`: Sparsity inducing penalty used during training. Can be choosen from `lasso`, `harder` or `scad`.

## How It Works
HarderLASSO combines:

1. **Neural Networks**: For capturing complex patterns in the data
2. **LASSO Regularization**: For encouraging sparsity in feature weights
3. **Quantile Universal Thresholding**: For determining optimal regularization parameters

The model trains to minimize both the prediction error and the number of selected features, resulting in a sparse, interpretable model.



