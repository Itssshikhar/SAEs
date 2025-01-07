# Adaptive Sparse Causal Attention for Character-Level Language Modeling

This project implements **Adaptive Sparse Causal Attention** (ASCA) for character-level language modeling on the **enwik8 dataset**. The research explores enhancing computational efficiency and performance by introducing content-based adaptive sparsity into attention mechanisms.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Implementation Details](#implementation-details)
  - [Content-Based Adaptive Sparsity Module](#content-based-adaptive-sparsity-module)
  - [Adaptive Sparse Causal Self-Attention](#adaptive-sparse-causal-self-attention)
- [Results and Discussion](#results-and-discussion)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Introduction

The **enwik8 dataset** is a widely recognized benchmark for evaluating compression capabilities in character-level language modeling. This project focuses on improving model efficiency and performance through a novel modification—**Adaptive Sparse Causal Attention**. 

### Goals
- Develop a character-level language model that captures sequential dependencies in data efficiently.
- Introduce a content-based sparsity mechanism to reduce computational burden.
- Demonstrate improved bits-per-character (BPC) performance over baseline transformer models.

---

## Key Features

- Implementation of **Content-Based Adaptive Sparsity Module** to dynamically prune attention weights.
- Integration with **Causal Self-Attention** while preserving autoregressive properties.
- Evaluation of model performance using BPC and sparsity metrics on the enwik8 dataset.

---

## Implementation Details

### Content-Based Adaptive Sparsity Module

This module predicts and applies a sparsity mask to prune irrelevant attention connections dynamically. Key features include:
- Content-based mask prediction via a small neural network.
- Dynamic adjustment of sparsity during training.
- Reduced computational overhead with efficient pruning.

```python
class ContentBasedAdaptiveSparsity(nn.Module):
    def forward(self, x, att):
        # Predict and apply sparsity mask
        ...
        return sparse_att
```

### Adaptive Sparse Causal Self-Attention

This module integrates adaptive sparsity into a standard causal self-attention mechanism:
- Ensures autoregressive properties with causal masking.
- Dynamically prunes attention connections based on input content.

```python
class AdaptiveSparseCausalSelfAttention(nn.Module):
    def forward(self, x, layer_past=None):
        # Apply causal attention with adaptive sparsity
        ...
        return y
```

---

## Results and Discussion

### Performance Metrics
- **Baseline Transformer Model**: BPC ~5.53 on enwik8 test set.
- **Adaptive Sparse Causal Attention Model**: BPC ~4.26, demonstrating better text compression with reduced computational cost.

### Sparsity Analysis
- Average sparsity: ~30% during training.
- Visualization shows dynamic adjustment of attention weights to retain critical connections.

### Key Benefits
- Improved BPC performance.
- Significant reduction in computational complexity.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Itssshikhar/SAEs.git
   cd SAEs
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the enwik8 dataset:**
   ```bash
   python prepare_dataset.py
   ```

---

## Usage

### Train Baseline Model:
```bash
python baseline_model.py
```

### Train Adaptive Sparse Model:
```bash
python novel_model.py
```

### Evaluate Results:
Use provided scripts to evaluate BPC and visualize sparsity.

---

## Project Structure

- `baseline_model.py`: Baseline transformer implementation.
- `novel_model.py`: Adaptive Sparse Causal Attention implementation.
- `prepare_dataset.py`: Script for preparing the enwik8 dataset.
- `configurator.py`: Configurations for models and training.
- `extraction.py`: Data extraction tools.
- `*.png` & `*.pdf`: Visualizations of sparsity and performance.

---

## References

1. [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/abs/1808.04444)

---

## License

This project is licensed under the [MIT License](LICENSE).
```

