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
Representation:Sequences were right-padded with 'N' to fixed lengths (28-nt for miRNA and 50-nt for the target site), concatenated into a 78-nt sequence, and one-hot encoded over the four nucleotides (A, C, G, T as binary indicators; 'N' padded positions as all zeros), resulting in 312 flattened features (78 × 4).

Model Architecture:A scikit-learn RandomForestClassifier with 100 decision trees (n_estimators=100, random_state=42) trained directly on the flattened one-hot feature vector (312 input features) to predict binary binding (label 0 vs. 1).

Validation & Evaluation Setup:Trained on the same 80/20 stratified split (6,512 training and 1,628 validation examples out of 8,140 official training pairs), using only noncodingRNA, gene, and label (excluding Nunique and holding out the official test set). Evaluated with ranking metrics (ROC AUC, Average Precision), threshold-dependent classification metrics at 0.5 (Accuracy, Balanced Accuracy, Precision, Recall, F1-score), and probability calibration (Log Loss).

Performance:On the validation split, the baseline Random Forest performed near chance (Accuracy ≈ 0.46, Balanced Accuracy ≈ 0.46, Precision ≈ 0.45, Recall ≈ 0.40, F1 ≈ 0.43, ROC AUC ≈ 0.45, Average Precision ≈ 0.50, Log Loss ≈ 0.72), lagging behind the MLP (ROC AUC ~0.72).

Limitation: Axis-aligned splits on 312 sparse, position-specific one-hot columns cannot easily capture cross-sequence base-pairing/complementarity (such as miRNA–target seed pairing) or shifted binding motifs across positions without engineered sequence/interaction features, leading to overfitting on training positions and weak generalization.


Conclusion
In this exercise, we built sequence-only classifiers to predict AGO2–CLASH miRNA–target binding (label 1 vs. 0) from miRBench (AGO2_CLASH_Hejret2023, v7, Zenodo 20540907) using strictly noncodingRNA, gene, and label—excluding Nunique and other metadata to avoid data leakage, and keeping the official chromosome-based test set isolated for final evaluation.

Comparing the two approaches on flattened position-level sequence representations highlights how model inductive bias shapes performance on raw sequence data:

Feed-forward neural network (MLP): Captured non-linear combinations across positions more effectively than axis-aligned tree splits, reaching roughly ~0.72 ROC AUC (~0.66 accuracy) on the test set, though flattened 1D inputs remain sensitive to motif shifts and show early overfitting (around Epoch 6) without explicit base-pairing or spatial structure.

Random Forest baseline: Trained on the 312-feature one-hot representation (78 padded positions across 28-nt miRNA and 50-nt target), it performed near chance on validation (ROC AUC ≈ 0.45, Average Precision ≈ 0.50), reflecting the difficulty of single-feature splits in capturing cross-sequence complementarity and positional shifts without domain-specific sequence features.

Overall, the experiment confirms that sequence-only models can learn partial binding signals under a chromosome-based holdout without leakage features, while explicit complementarity/alignment cues, convolutional or attention-based architectures, and engineered sequence features are natural next steps to improve generalization.
