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

