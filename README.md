# EEG-Functional-Connectivity-Graph-Deep-Learning
Graph-based deep learning framework for EEG functional connectivity analysis and brain state classification (Eyes Open vs Eyes Closed) using machine learning, graph neural networks, and statistical validation

## Overview
This repository presents an end-to-end computational neuroscience project focused on the analysis of EEG functional connectivity using both classical Machine Learning (ML) and Graph Neural Networks (GNNs).
Rather than treating EEG recordings as independent time series, this project models the brain as a complex network, where brain regions (electrodes) ares represented as graph nodes and functional connections between them as graph edges.
The objective is to investigate whether graph-based deep learning methods can effectively classify brain states from EEG functional connectivity networks and provide a reproducible computational pipeline for neuroscience research.

## Research Objectives
- Build functional connectivity networks from EEG recordings.
- Represent EEG data as graphs using PyTorch Geometric.
- Analyze brain connectivity through graph-based representations.
- Develop baseline Machine Learning models.
- Implement Graph Convolutional Networks (GCNs) for graph classification.
- Compare classical Machine Learning and Graph Neural Networks.
- Evaluate model performance using stratified 5-fold cross-validation.

## Project Pipeline
EEG Signals
      │
      ▼
Data Preprocessing
      │
      ▼
Functional Connectivity Matrix
      │
      ▼
Brain Graph Construction
      │
      ▼
Graph Analysis
      │
      │──────────────► Classical Machine Learning
      │
Graph Neural Networks (GCN)
      │
      ▼
Model Evaluation

## Data Source
The EEG data used in this project were obtained from the OpenNeuro repository.
**OpenNeuro:** ds004148
https://openneuro.org/datasets/ds004148/versions/1.0.1
The dataset was preprocessed to construct functional connectivity graphs suitable for graph-based deep learning using PyTorch Geometric.
All credit for the original data collection belogs to the autors of the OpenNeuro dataset.

## Dataset Representation
Each EEG recording is converted into a graph.
- **Nodes:** EEG electrodes (61 channels)
- **Edges:** Functional connectivity between electrodes
- **Node Features:** Connectivity-derived features
- **Graph Label:** Brain state class

The graphs are stored using PyTorch Geometric 'Data' object.

## Graph Neural Network Architecture
The implemented Graph Convolutional Network (GCN)  consists of:

Input Graph
      │
GCNConv (61 → 32)
      │
ReLU
      │
GCNConv (32 → 16)
      │
ReLU
      │
Global Mean Pooling
      │
Linear Layer
      │
Brain State Prediction

## Model Evaluation:
Models are evaluated using
- Stratified 5-fold cross validation
- CrossEntropy Loss
- Adam Optimizer
- Accuracy as the primary evaluation metric

The final performances is reported as:
- Mean Accuracy
- Standard Deviation

## Technologies
- Python
- PyTorch
- PyTorch Geometric
- Numpy
- Pandas
- Scikit-learn
- Matplotlib

## Research Motivation
Brain activity is inherently network-based.
Instead of analysing individual EEG channels independently, this project studies how brain regions interact through functional connectivity networks.
Graph Neural Networks provide a natural framework for learning directly from these connectivity patterns.

## Future Work
- Graph Attention Networks (GAT)
- GraphSAGE
- Dynamic Functional Connectivity
- Explainable AI for Brain Networks
- Larger EEG datasets
- Clinical applications in neurological disorders
- Integration with multimodal neuroimaging data





     
