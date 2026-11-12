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