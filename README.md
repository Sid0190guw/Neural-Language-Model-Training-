# Neural-Language-Model-Training-
Train a neural language model from scratch using PyTorch. The goal is to demonstrate  understanding of how sequence models learn to predict text and how model design and  training affect performance. 
# Neural Language Model - Smoke Test Notebook

This Jupyter notebook contains a compact LSTM language-model smoke test designed for Google Colab.

## How to run on Colab
1. Upload the notebook to Google Drive or open it directly in Colab.
2. **Runtime → Change runtime type → GPU** (optional but recommended).
3. Run the first cell. When prompted, upload `Pride_and_Prejudice-Jane_Austen.txt` (public-domain text).
4. The script trains three quick regimes (underfit / overfit / bestfit) and saves outputs to `/content`.
   - To save to Drive, uncomment the drive mount lines and set `DRIVE_SAVE` accordingly.

## Files produced
- `*_smoke.pt` — model checkpoints
- `*_smoke_loss.png` — loss plots

📘 Neural Language Model Training (PyTorch)

This project implements a Neural Language Model trained from scratch using PyTorch, as required in Assignment 2: Neural Language Model Training.
The goal is to demonstrate understanding of sequence models, loss behavior, perplexity, and generalization regimes (underfitting, overfitting, best-fit).

🧠 Objective

Build a word-level Language Model using PyTorch (no pre-trained models).

Train and evaluate:

Underfitting configuration

Overfitting configuration

Best-fit configuration

Generate Training vs Validation Loss curves.

Compute perplexity (PPL) as evaluation metric.

Compare the three regimes and justify the best-fit model.

📂 Project Structure
📁 project/
│
├── data/                     # Preprocessed dataset (Pride and Prejudice)
│
├── models/                   # Saved PyTorch model checkpoints
│   ├── underfit.pt
│   ├── overfit.pt
│   └── bestfit.pt
│
├── loss_plots/               # Loss curves saved as PNG & PDF
│   ├── underfit_train_val.png
│   ├── overfit_train_val.png
│   ├── bestfit_train_val.png
│   ├── underfit_train_losses.npy
│   └── ...
│
├── Neural.ipynb              # Main Colab notebook
│
├── train.py                  # Script version (optional)
│
└── README.md                 # (This file)

📚 Dataset

The dataset used is:

📘 Pride and Prejudice — Jane Austen

Preprocessing steps:

Lowercasing

Replace newline \n with <nl>

Word-level tokenization (text.split())

Keep top 8000 most frequent words

Rare/unseen words mapped to <unk>

Sequence length = 30 tokens

Split:

90% training

10% validation

Total tokens after preprocessing: 138,682

🏗️ Model Architecture (LSTM Language Model)

Embedding layer

1–3 LSTM layers (depending on config)

Dropout (0.0 to 0.2)

Fully-connected layer → vocab logits

Softmax

Loss: CrossEntropyLoss

Optimizer: Adam

Metric: Perplexity = exp(validation_loss)

⚙️ Training Configurations
🔴 Underfitting
Parameter	Value
Embedding	32
Hidden Size	64
LSTM Layers	1
Dropout	0.2
Epochs	2
Batch size	128
LR	1e-3
🔵 Overfitting
Parameter	Value
Embedding	128
Hidden Size	256
LSTM Layers	2
Dropout	0.0
Epochs	3
Batch size	64
LR	1e-3
🟢 Best-Fit
Parameter	Value
Embedding	128
Hidden Size	256
LSTM Layers	2
Dropout	0.2
Epochs	3
Batch size	64
LR	1e-3
📊 Results
✔️ Final Perplexity
Model	Validation Loss	Perplexity
Underfit	5.6870	295.01
Overfit	5.1616	174.45
Best-Fit	5.2672	193.87
📉 Training & Validation Loss Curves

All loss curves are located in:

loss_plots/


Including:

underfit_loss.png

overfit_loss.png

bestfit_loss.png

.npy files containing raw loss values

🚀 How to Run the Notebook (Google Colab)
1️⃣ Open the notebook

Click below (ensure you are logged into Google Drive):

➡️ Neural.ipynb

2️⃣ Install dependencies

(If needed, inside Colab)

pip install torch numpy matplotlib tqdm

3️⃣ Run preprocessing

This loads the dataset, tokenizes it, builds vocabulary, and prepares dataloaders.

4️⃣ Train models

Run the training cells:

train_underfit = train_and_record(...)

train_overfit = train_and_record(...)

train_bestfit = train_and_record(...)

5️⃣ View curves

Plots will automatically appear and will also be saved into loss_plots/.

🏁 Conclusion

The assignment demonstrates:

Underfitting → small model, high bias

Overfitting → strong memorization, poor generalization

Best-Fit → balanced performance and stability

The best-fit LSTM model is selected because it achieves the best trade-off between perplexity and generalization.
