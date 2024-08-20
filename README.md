# Neural-Importance-Score-Propagation

This repository includes the PyTorch implementation of [Neural Importance Score Propagation (NISP)](https://arxiv.org/abs/1711.05908) <br/> 
NISP is a post-pruning method that reduces the size of a neural network by considering its global structure. This structured pruning approach calculates the pruning score based on the derivative of the loss function with respect to the output activation of each intermediate layer.

`NISPPruner.py`
- Contains the code for the pruning logic. <br/>

`pruning_test.ipynb`
- Script for training ResNet50 on the CIFAR10 dataset, followed by using NISPPruner to prune the resulting model artifact.
