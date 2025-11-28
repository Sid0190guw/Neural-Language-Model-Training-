# Neural Language Model Training – Assignment 2
**Research Internship Evaluation – IIIT Hyderabad**

This repository contains the complete submission for **Assignment 2: Neural Language Model Training (PyTorch)** as part of the Research Internship evaluation at IIIT Hyderabad.

It includes:
- Full PyTorch implementation of a word-level language model
- Data preprocessing utilities
- Training pipeline
- Three experimental regimes: underfitting, overfitting, and best-fit
- Loss curves and evaluation metrics
- (Optional) model checkpoints stored externally

---

## Project Overview

### Objective
To design, train, and analyse a word-level neural language model using PyTorch and demonstrate:
- Underfitting
- Overfitting
- A well-generalized (best-fit) model

### Dataset
- Pride and Prejudice (provided dataset)
- Word-level tokenization
- Vocabulary limited to 8000 tokens

---

## Model Architecture

A standard LSTM-based language model:

**Embedding Layer → LSTM → Fully Connected Linear Layer**

Configurable parameters:
- Embedding dimension
- Hidden dimension
- Number of layers

Training:
- Loss: Cross-entropy
- Metric: Perplexity (PPL)

---

## Repository Structure

```
├── data_utils.py               # Data processing & vocabulary creation
├── model.py                    # LSTM Language Model
├── train.py                    # Training & evaluation script
├── Assignment2_Complete_Report.pdf
├── checkpoints/                # (Optional) External storage
└── README.md
```

---

## Running the Project

### 1. Install Dependencies
```bash
pip install torch numpy tqdm
```

### 2. Train the Model
```bash
python train.py
```

### 3. Example Experimental Settings

**Underfitting**
```bash
python train.py --embed 32 --hidden 32 --epochs 3 --lr 0.01
```

**Overfitting**
```bash
python train.py --embed 256 --hidden 512 --epochs 50 --lr 0.001
```

**Best-fit**
```bash
python train.py --embed 128 --hidden 256 --epochs 20 --lr 0.001
```

---

## Results Summary

| Experiment  | Perplexity |
|-------------|------------|
| Underfit    | 295.01     |
| Overfit     | 174.45     |
| Best-fit    | 193.87     |

Detailed loss curves and analysis are included in:
- Assignment2_Complete_Report.pdf
- External Drive links for plots and checkpoints

---

## Checkpoints and Loss Plots

Trained checkpoints and loss curve images are available in the linked Google Drive folder.

---

## Reproducibility

- Fixed random seeds
- CPU/GPU compatible instructions
- Scripts tested on Google Colab

---

## Detailed Report

The provided report includes:
- Objective and dataset description
- Preprocessing steps
- Model architecture explanation
- Hyperparameters
- Loss curves
- Behaviour analysis (underfit, overfit, best-fit)
- Perplexity comparisons
- Notes on deterministic dataloaders

---

# Neural Language Model – Smoke Test Notebook

A compact LSTM-based smoke test notebook is provided for Google Colab.

### Running on Colab
1. Open the notebook in Google Colab.
2. (Optional) Enable GPU from the runtime menu.
3. Upload the dataset text file when prompted.
4. The notebook trains underfit, overfit, and best-fit models.
5. Outputs are saved in `/content` unless Drive saving is enabled.

### Files Produced
- `*_smoke.pt` — model checkpoints
- `*_smoke_loss.png` — loss plots

---

# Assignment Objective Summary

The goal is to implement a word-level LSTM language model from scratch and evaluate how model capacity and training behaviour influence:
- Bias (underfitting)
- Variance (overfitting)
- Generalization (best-fit)

Metrics such as training loss, validation loss, and perplexity guide model comparison.

---

# Dataset Details

- Source: *Pride and Prejudice* by Jane Austen
- Preprocessing includes:
  - Lowercasing
  - Tokenization
  - Newline markers
  - Vocabulary limiting
  - Replacing rare words with `<unk>`
  - Forming sequences of length 30 tokens
  - Train/validation split

Final token count: 138,682

---

# Model Details

- Embedding layer
- 1–3 LSTM layers (experiment dependent)
- Optional dropout
- Fully connected output layer
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Metric: Perplexity = exp(validation_loss)

---

# Training Configurations

## Underfitting
- Embedding: 32  
- Hidden size: 64  
- Layers: 1  
- Dropout: 0.2  
- Epochs: 2  
- Batch size: 128  
- Learning rate: 1e-3  

## Overfitting
- Embedding: 128  
- Hidden size: 256  
- Layers: 2  
- Dropout: 0.0  
- Epochs: 3  
- Batch size: 64  
- Learning rate: 1e-3  

## Best-Fit
- Embedding: 128  
- Hidden size: 256  
- Layers: 2  
- Dropout: 0.2  
- Epochs: 3  
- Batch size: 64  
- Learning rate: 1e-3  

---

# Results

| Model      | Validation Loss | Perplexity |
|------------|------------------|------------|
| Underfit   | 5.6870           | 295.01     |
| Overfit    | 5.1616           | 174.45     |
| Best-fit   | 5.2672           | 193.87     |

Loss plots are located in the `loss_plots/` directory.

---

# Conclusion

The assignment demonstrates core concepts of neural language modeling:
- Underfitting results from limited model capacity.
- Overfitting results from excessive model capacity and memorization.
- A best-fit model balances bias and variance.

The best-fit configuration offers the most stable generalization performance based on perplexity and loss behaviour.
