import numpy as np
import pandas as pd
import torch
import torch.nn as nnpip
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from copy import deepcopy

seed=42
max_epochs=15
batch_size=64
learning_rate=0.001
mirna_length=28
target_length=50
output_dir=Path('project_output')
combined_len=mirna_length+target_length  #78
#why did not use threshold?
#why need combined lenght?
#why not use torch.set_num_threads(2)?

torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


##### Load training data from Zenodo #####
data_url='https://zenodo.org/records/20540907/files/'
input_columns=['noncodingRNA', 'gene', 'label']

benchmark_train=pd.read_csv(data_url+'AGO2_CLASH_Hejret2023_train.tsv.gz?download=1', sep='	', compression='gzip', usecols=input_columns,)

'''
Why do we not need all these?
assert benchmark_train['noncodingRNA'].str.fullmatch(f'[ACGT]{{1,{mirna_length}}}').all()
assert benchmark_train['gene'].str.fullmatch(f'[ACGT]{{{target_length}}}').all()
assert set(benchmark_train['label']) == {0, 1}
assert not benchmark_train.duplicated(['noncodingRNA', 'gene']).any()
print('Official training examples:', len(benchmark_train))
print('miRNA lengths:', benchmark_train['noncodingRNA'].str.len().min(),
      'to', benchmark_train['noncodingRNA'].str.len().max())
print(benchmark_train['label'].value_counts())
benchmark_train.head()
'''

train_data, val_data=train_test_split(benchmark_train, test_size=0.20, stratify=benchmark_train['label'], random_state=seed)
print(f"Training pairs: {len(train_data)} | Validation pairs: {len(val_data)}")

'''
Why not use these?
validation_data = benchmark_train.groupby('label').sample(frac=0.20, random_state=seed)
train_data = benchmark_train.drop(validation_data.index)
y_train = train_data['label'].to_numpy()
y_validation = validation_data['label'].to_numpy()
print('Training:', len(train_data))
print('Validation:', len(validation_data))
'''

##### Sequence Tokenazation #####
def encode_pairs(data):
    base_ids={'A':0, 'C':1, 'G':2, 'T':3}
    codes=np.zeros((len(data), combined_len), dtype=np.int64)
    labels=data['label'].to_numpy(dtype=np.float32)

    for i, (mirna, target) in enumerate(zip(data['noncodingRNA'], data['gene'])):
        print()