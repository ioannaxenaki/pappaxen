# pappaxen
Exercise 10: miRNA–target prediction group project

To resolve the torch problem do this inside the pappaxen folder from the terminal:
python -m pip install -r requirements.txt



MLP Approach (Xenaki Ioanna)

Representation: MLPs require fixed-length 1D inputs. We mapped nucleotides to integers, right-padded miRNAs to 28-nt (using a PAD token), and concatenated them with the 50-nt target. This 78-nt sequence was one-hot encoded and flattened into exactly 390 input features.
Model Architecture: A PyTorch feed-forward network funneling from 390 → 128 → 64 → 1 logit.
Justification & Regularization: To prevent memorization of absolute positions, we used ReLU activations, 30% Dropout on hidden layers, and L2 Regularization (weight decay = 0.0001) via the Adam optimizer.

The model strickly uses an 80/20 stratified validation split for epoch/hyperparameter selection before final testing.
Primary Metrics (Ranking): ROC AUC and Average Precision evaluate global ranking capability across all probability thresholds.
Secondary Metrics (Classification): Precision, Recall, F1-score, Accuracy, and Balanced Accuracy evaluate strict binary classification at a 0.5 threshold. Log Loss measures probability calibration.

Performance: The MLP achieved an ROC AUC of ~0.72 and an accuracy of ~0.66 on the official test set.
Limitation: Flattening sequences into 1D vectors destroys spatial invariance. Unlike CNNs, if a biological binding motif shifts by one nucleotide, the MLP treats it as entirely new, unrecognizable data. This strict positional dependence caused rapid overfitting (optimal validation loss hit at just Epoch 6).



Random Forest (Pappa Eleftheria)