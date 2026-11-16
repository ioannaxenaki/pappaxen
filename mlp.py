import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score,recall_score, f1_score, roc_auc_score, average_precision_score, log_loss, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from pathlib import Path
from copy import deepcopy

seed=42
max_epochs=15
batch_size=64
learning_rate=0.001
mirna_length=28
target_length=50
output_dir=Path('MLP_output')
threshold=0.5

torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


##### Load training data from Zenodo #####
data_url='https://zenodo.org/records/20540907/files/'
input_columns=['noncodingRNA', 'gene', 'label']
benchmark_train=pd.read_csv(data_url+'AGO2_CLASH_Hejret2023_train.tsv.gz?download=1', sep='	', compression='gzip', usecols=input_columns,)



##### Validation Split #####
validation_data=benchmark_train.groupby('label').sample(frac=0.20, random_state=seed)
train_data=benchmark_train.drop(validation_data.index)
y_train=train_data['label'].to_numpy()
y_validation=validation_data['label'].to_numpy()
print('Training:', len(train_data))
print('Validation:', len(validation_data))



##### Represent sequences as 1D vectors #####
#MLP needs a fixed lengthlist of numbers
#We assign each nucleotide a number and since miRNAs vary in length, we pad them to 28 characters using a "PAD" token
#The target is always 50 characters, so we combine them to a single sequence of 78 characters
#We one-hot encode this sequence and flatten it. 78 positions * 5 categories = 390 input features

def encode_flat_sequences(data):
    base_ids={'A':0, 'C':1, 'G':2, 'T':3}
    pad=4
    encoded_list=[]

    for mirna, target in zip(data['noncodingRNA'], data['gene']):
        #Convert letters to numbers
        mirnas_nums=[base_ids[i] for i in mirna]
        target_nums=[base_ids[i] for i in target]

        #Pad miRNAs to 28 length
        mirnas_nums+=[pad]*(mirna_length-len(mirnas_nums))

        #Combine miRNA and target to 78 total legth
        combined=mirnas_nums+target_nums
        encoded_list.append(combined)

    #Convert to tensor
    tensor_seqs=torch.tensor(encoded_list, dtype=torch.long)

    #One-hot encode creates a shape of [N, 78, 5]
    one_hot=F.one_hot(tensor_seqs, num_classes=5).float()

    #Flatten to [N, 390] for the MLP
    flat_vectors=one_hot.view(len(data), -1)

    return flat_vectors

#Encode the datasets
X_train=encode_flat_sequences(train_data)
X_val=encode_flat_sequences(validation_data)

#Create PyTorch DataLoaders
train_loader=DataLoader(TensorDataset(X_train, torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
val_loader=DataLoader(TensorDataset(X_val, torch.tensor(y_validation, dtype=torch.float32)), batch_size=batch_size, shuffle=False)



##### Train the MLP model #####
#A feed-forward neural network that takes 390 features in
#It passes them through a hidden layer with ReLu activation and outputs a single value indicating the likelihood of a binding pair
#Dropout randomly turns off neurons during training, which prevents the network from memorizing the training data

mlp=nn.Sequential(
    nn.Linear(390, 128),    #Input layer: 390 features to 128 hidden nodes
    nn.ReLU(),  #Activation function
    nn.Dropout(0.3),    #30% dropout to prevent overfitting
    
    nn.Linear(128, 64), #Hidden layer: 128 nodes to 64 nodes
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(64, 1)    #Output layer: 1 node
).to(device)

loss_function=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=0.0001)

print(mlp)
print('Trainable parameters:', sum(p.numel() for p in mlp.parameters() if p.requires_grad))



##### Training and Epoch Selection #####
train_losses=[]
val_losses=[]
best_val_loss=np.inf
best_state=None
best_epoch=0

for epoch in range(1, max_epochs+1):
    #Train
    mlp.train()
    total_train_loss=0
    for inputs, labels in train_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        logits=mlp(inputs).squeeze(1)
        loss=loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_train_loss+=loss.item() * len(labels)
    
    average_train_loss=total_train_loss / len(train_data)

    #Validation
    mlp.eval()
    total_val_loss=0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            logits=mlp(inputs).squeeze(1)
            loss=loss_function(logits, labels)
            total_val_loss+=loss.item() * len(labels)

    average_val_loss=total_val_loss / len(validation_data)

    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)

    #Save best model
    if average_val_loss<best_val_loss:
        best_val_loss=average_val_loss
        best_epoch=epoch
        best_state=deepcopy(mlp.state_dict())

    print(f'Epoch {epoch}: Train loss={average_train_loss:.4f}, Validation loss={average_val_loss:.4f}', flush=True)

#Restore best weights
mlp.load_state_dict(best_state)
print(f"\nSelected best model from Epoch {best_epoch}")

#Plot the training progress
plt.figure(figsize=(8, 4))
plt.plot(range(1, max_epochs+1), train_losses, label='Training (with dropout)')
plt.plot(range(1, max_epochs+1), val_losses, label='Validation')
plt.axvline(best_epoch, color='black', linestyle='--', label='Selected epoch')
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy Loss')
plt.legend()
plt.title('MLP Training Progress')
plt.tight_layout()
plt.show()



##### Evaluate Validation Predictions #####
def predict_probabilities(model, loader):
    model.eval()
    batches=[]
    with torch.no_grad():
        for inputs, _ in loader:
            logits=model(inputs.to(device)).squeeze(1)
            batches.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(batches)

def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= threshold).astype(int)
    return pd.Series({
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(labels, probabilities),
        'average_precision': average_precision_score(labels, probabilities),
        'log_loss': log_loss(labels, probabilities),
    })

def plot_evaluation(labels, probabilities, name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    RocCurveDisplay.from_predictions(labels, probabilities, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], 'k--')
    PrecisionRecallDisplay.from_predictions(labels, probabilities, ax=axes[1])
    axes[1].axhline(np.mean(labels), color='black', linestyle='--')
    ConfusionMatrixDisplay.from_predictions(labels, probabilities >= threshold,
                                           labels=[0, 1], ax=axes[2], cmap='Blues', colorbar=False)
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()

validation_probabilities=predict_probabilities(mlp, val_loader)
validation_results=calculate_metrics(y_validation, validation_probabilities)
print(validation_results.round(4).to_string())
plot_evaluation(y_validation, validation_probabilities, 'Validation: MLP')
output_dir.mkdir(parents=True, exist_ok=True)
validation_ids='train_' + validation_data.index.astype(str)
pd.DataFrame({'sample_id': validation_ids, 'probability': validation_probabilities}).to_csv(output_dir / 'validation_predictions.csv', index=False)
pd.DataFrame({'sample_id': validation_ids, 'label': y_validation}).to_csv(output_dir / 'validation_answers.csv', index=False)



##### Final Evaluation on Test Set #####
test_data=pd.read_csv(data_url + 'AGO2_CLASH_Hejret2023_test.tsv.gz?download=1', sep='	', compression='gzip', usecols=input_columns,)
X_test=encode_flat_sequences(test_data)
y_test=test_data['label'].to_numpy()
test_loader=DataLoader(TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
test_probabilities=predict_probabilities(mlp, test_loader)
test_results=calculate_metrics(y_test, test_probabilities)
print('Official test examples:', len(test_data))
print(test_results.round(4).to_string())
plot_evaluation(y_test, test_probabilities, 'Official test: MLP')
test_ids='test_' + test_data.index.astype(str)
pd.DataFrame({'sample_id': test_ids, 'probability': test_probabilities}).to_csv(output_dir / 'test_predictions.csv', index=False)
pd.DataFrame({'sample_id': test_ids, 'label': y_test}).to_csv(output_dir / 'test_answers.csv', index=False)
print('Prediction and answer files saved in:', output_dir)