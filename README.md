# EEG Functional Connectivity Analysis Using Machine Learning and Graph Neural Networks
An end-to-end computational neuroscience pipeline for brain-state classification (eyes open vs. eyes closed) using functional connectivity, graph theory, machine learning, and Graph Neural Networks.

## Project Summary

| Item | Description |
|------|-------------|
| **Problem** | Brain state classification from EEG functional connectivity |
| **Dataset** | OpenNeuro ds004148 |
| **Conditions** | Eyes Open vs. Eyes Closed |
| **Graphs** | 120 functional connectivity graphs |
| **Nodes per graph** | 61 EEG electrodes |
| **Framework** | PyTorch Geometric |
| **Graph Model** | Graph Convolutional Network (GCN) |
| **Validation** | Stratified 5-Fold Cross Validation |
| **Best Accuracy** | 72.50% (Logistic Regression) |
| **GCN Accuracy** | 69.17% |

## Overview
This repository presents an end-to-end computational neuroscience project focused on the analysis of EEG functional connectivity using both classical Machine Learning (ML) and Graph Neural Networks (GNNs).
Rather than treating EEG recordings as independent time series, this project models the brain as a complex network, where brain regions (electrodes) are represented as graph nodes and functional connections between them as graph edges.

While previous analyses relied on handcrafted graph-theoretical features, this study investigates whether representation learning through Graph Neural Networks can capture additional information from EEG functional connectivity networks for brain-state classification.

The **objective** is to investigate whether graph-based deep learning methods (Graph Neural Networks) can improve EEG-brain state classification by learning directly from functional connectivity graphs and to compare their performance against classical machine learning models.

## Research Question
Can Graph Neural Networks learn discriminative representations from EEG functional connectivity graphs and outperform classical graph-theoretical feature-based machine learning models in eyes-open versus eyes-closed brain state classification?

## Hypothesis
Graph Neural Networks will achieve higher classification performance than classical machine learning models trained on handcrafted graph-theoretical features because they can directly learn topological patterns from EEG functional connectivity networks.

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
All credit for the original data collection belongs to the authors of the OpenNeuro dataset.

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

The final performance is reported as:
- Mean Accuracy
- Standard Deviation
  
## Results
### Significant Functional Connections (FDR)
Fig 1. significant_functional_connections_fdr.png
- The heatmap highlights the statistically significant functional connections after applying False Discovery Rate (FDR) correction. In the figure, each significant point represents a functional connection that remained statistically significant after multiple-comparison-correction.

### Brain Functional Network (FDR)
Fig 2. brain_functional_network_fdr.png
- In the visualization of the functional brain network constructed from significant connections after FDR correction, the nodes represent EEG electrodes.Edges represent significant functional interactions between brain regions.

### Brain Functional Network - Closed Eyes and Open Eyes
Fig 3. brain_network_closed_eyes_fdr.png
Fig 4. brain_network_open_eyes_fdr.png
- The graph illustrates the organization of significant functional connections between EEG electrodes after FDR correction.

### Three-Dimensional Brain Connectivity Network
Fig 5. brain_connectivity_network_3d.png
- The nodes correspond to EEG electrode locations projected onto a  3D head model, while edges represent significant functional connections between brain regions.

### Model Performance Comparison
Fig 6. compare_models_accuracy.png
- In the picture, the error bars indicate variability across cross-validation folds.
- The compared models are: logistic regression, support vector machine, random forest, k-nearest neighbors, graph convolutional network.

### Classification Performance

| Model | Accuracy (%) | Notes |
|:--------------------------|:------------:|:-------------------------------------------------------------|
| Logistic Regression | **72.50** | Highest classification accuracy among all evaluated models. |
| Support Vector Machine (SVM) | 70.50 | Highest AUC (0.795) and robust cross-validation performance. |
| Graph Convolutional Network (GCN) | 69.17 | Learned directly from graph topology without handcrafted features. |
| Random Forest | 68.33 | Assortativity was identified as the most important graph feature. |
| K-Nearest Neighbors (KNN) | 63.33 | Lowest classification performance in this study. |

### Cross-Validated ROC Curve
Fig 7. cross_validated_ROC_curve.png
- The Receiver Operating Characteristic (ROC) curve was obtained using stratified cross-validation. The mean AUC for cross-validated SVM was 0.7951.

### Connectivity Statistics
Fig 8. connectivity_statistics.png
- The figure includes t-statistics, p-values and significant connections after thresholding (p<0.05).

### Functional Connectivity Heatmaps
Fig 9. connectivity_heatmaps.png
- The figure represents mean functional connectivity matrices computed for each experimental condition (mean connectivity eyes open, mean connectivity eyes closed, difference matrix closed vs open). The objective of this connectivity networks was to identify state-dependent reconfiguration of brain connectivity.

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

## Discussion
- The results indicate that classical machine learning and Graph Neural Network approaches achieved comparable performance for distinguishing eyes-open and eyes-closed EEG functional connectivity states.
- Although the Graph Convolutional Network did not outperform the best classical model (Logistic Regression), it successfully learned directly from graph topology without relying on handcrafted graph-theoretical features.
- One possible explanation is the relatively small dataset used in this study (60 subjects, 120 graphs), which may limit the representation learning capacity of deep learning models. Graph Neural Networks generally benefit from substantially larger graph datasets.
- For the Random Forest classifier, feature importance analysis revealed that **assortativity** was the most informative graph-theoretical metric (importance = 0.2566), suggesting that changes in the connectivity patterns between highly connected nodes contribute to differentiating the eyes-open and eyes-closed conditions.
  
## Interpretation
- Classical machine learning and graph neural network approaches achieved comparable performance in discriminating eyes-open and eyes-closed EEG functional connectivity states.
- Also, the differences between open and closed eyes are concentrated in parieto-occipital regions associated with visual processing.
- Classical models achieved 68–73% accuracy, while the GCN achieved 69%, which suggests that global network metrics contain discriminative information comparable to that learned directly from the graph structure.
  
## Future Work
- Graph Attention Networks (GAT)
- GraphSAGE
- Dynamic Functional Connectivity
- Explainable AI for Brain Networks
- Larger EEG datasets
- Clinical applications in neurological disorders
- Integration with multimodal neuroimaging data





     
