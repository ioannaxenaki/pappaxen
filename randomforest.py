from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, log_loss,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier

seed = 42
max_epochs = 15
batch_size = 64
learning_rate = 0.001
threshold = 0.5
mirna_length = 28
target_length = 50
output_dir = Path('exercise_10_output')

torch.manual_seed(seed)
torch.set_num_threads(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

data_url = 'https://zenodo.org/records/20540907/files/'
input_columns = ['noncodingRNA', 'gene', 'label']
benchmark_train = pd.read_csv(
    data_url + 'AGO2_CLASH_Hejret2023_train.tsv.gz?download=1',
    sep='	', compression='gzip', usecols=input_columns,
)
assert benchmark_train['noncodingRNA'].str.fullmatch(f'[ACGT]{{1,{mirna_length}}}').all()
assert benchmark_train['gene'].str.fullmatch(f'[ACGT]{{{target_length}}}').all()
assert set(benchmark_train['label']) == {0, 1}
assert not benchmark_train.duplicated(['noncodingRNA', 'gene']).any()
print('Official training examples:', len(benchmark_train))
print('miRNA lengths:', benchmark_train['noncodingRNA'].str.len().min(),
      'to', benchmark_train['noncodingRNA'].str.len().max())
print(benchmark_train['label'].value_counts())
benchmark_train.head()

validation_data = benchmark_train.groupby('label').sample(frac=0.20, random_state=seed)
train_data = benchmark_train.drop(validation_data.index)
y_train = train_data['label'].to_numpy()
y_validation = validation_data['label'].to_numpy()
print('Training:', len(train_data))
print('Validation:', len(validation_data))

# Pad sequences to a fixed length
def pad_sequence(sequence, max_length):
    return sequence.ljust(max_length, 'N')

# Convert miRNA and gene sequences to numerical features
def encode_sequence(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return [mapping[base] for base in sequence]

# Pad and encode sequences
train_data['padded_mirna'] = train_data['noncodingRNA'].apply(pad_sequence, args=(mirna_length,))
train_data['padded_gene'] = train_data['gene'].apply(pad_sequence, args=(target_length,))
validation_data['padded_mirna'] = validation_data['noncodingRNA'].apply(pad_sequence, args=(mirna_length,))
validation_data['padded_gene'] = validation_data['gene'].apply(pad_sequence, args=(target_length,))

X_train_mirna = np.array([encode_sequence(seq) for seq in train_data['padded_mirna']])
X_train_gene = np.array([encode_sequence(seq) for seq in train_data['padded_gene']])
X_train = np.concatenate((X_train_mirna, X_train_gene), axis=1)

X_validation_mirna = np.array([encode_sequence(seq) for seq in validation_data['padded_mirna']])
X_validation_gene = np.array([encode_sequence(seq) for seq in validation_data['padded_gene']])
X_validation = np.concatenate((X_validation_mirna, X_validation_gene), axis=1)

# Create and train the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
rf_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_rf = rf_classifier.predict(X_validation)
y_pred_proba_rf = rf_classifier.predict_proba(X_validation)[:, 1]

# Evaluate the random forest model
print("Random Forest Metrics:")
print("Accuracy:", accuracy_score(y_validation, y_pred_rf))
print("Balanced Accuracy:", balanced_accuracy_score(y_validation, y_pred_rf))
print("Precision:", precision_score(y_validation, y_pred_rf))
print("Recall:", recall_score(y_validation, y_pred_rf))
print("F1 Score:", f1_score(y_validation, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_validation, y_pred_proba_rf))
print("Average Precision Score:", average_precision_score(y_validation, y_pred_proba_rf))
print("Log Loss:", log_loss(y_validation, y_pred_proba_rf))