from sys import stderr
import numpy as np
from pandas import read_csv

import torch
from torch import nn, from_numpy, sigmoid, squeeze
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score

log = stderr.write

trainall_sz = 592380
train_sz = 500000

class NT:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

badcols = ['opened_position_qty ', 'closed_position_qty']
df_train_all = read_csv('train.csv').drop(columns=badcols+['id'])
df_test = read_csv('test.csv').drop(columns=badcols)

def add_cols(df):
    for j in range(5):
        i = j+1
        df[f'bid{i}mul'] = df[f'bid{i}']*df[f'bid{i}vol']
        df[f'ask{i}mul'] = df[f'ask{i}']*df[f'ask{i}vol']
add_cols(df_train_all)
add_cols(df_test)

for c in df_test.columns:
    c = str(c)
    if c == 'id':
        continue
    m = np.max(np.concatenate((
        df_train_all[c].to_numpy(),
        df_test[c].to_numpy())))
    df_train_all[c] /= m
    df_test[c] /= m

np.random.seed(13880)
perm = np.random.permutation(trainall_sz)
df_train = df_train_all.iloc[perm[:train_sz]]
df_val = df_train_all.iloc[perm[train_sz:]]

train = NT(
        x = from_numpy(df_train.drop(columns='y').to_numpy()).double(),
        y = from_numpy(df_train['y'].to_numpy(dtype=np.double)))
val = NT(
        x = from_numpy(df_val.drop(columns='y').to_numpy()).double(),
        y = df_val['y'].to_numpy(dtype=np.double))
test = NT(
        id = df_test['id'].to_numpy(),
        x = from_numpy(df_test.drop(columns='id').to_numpy()).double())

# includes the label in the data to make sure that the algorithm is
# actually running correctly.
'''
train = NT(
        x = from_numpy(df_train.to_numpy()).double(),
        y = from_numpy(df_train['y'].to_numpy(dtype=np.double)))
val = NT(
        x = from_numpy(df_val.to_numpy()).double(),
        y = df_val['y'].to_numpy(dtype=np.double))
'''

train_loader = DataLoader(
        TensorDataset(train.x, train.y),
        batch_size=1024, shuffle=True)

model = nn.Sequential(
        nn.Linear(34, 30),
        nn.PReLU(),

        nn.Linear(30, 30),
        nn.PReLU(),

        nn.Linear(30, 30),
        nn.PReLU(),

        nn.Linear(30, 30),
        nn.PReLU(),

        nn.Dropout(.15),
        nn.Linear(30, 1))
model.double()

crit = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-2)

for e in range(10):
    model.train()
    for x, y in train_loader:
        optim.zero_grad()
        loss = crit(squeeze(model(x)), y)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        vscore = roc_auc_score(
                val.y,
                sigmoid(squeeze(model(val.x))).numpy())
        tscore = roc_auc_score(
                train.y.numpy(),
                sigmoid(squeeze(model(train.x))).numpy())
        loss = crit(squeeze(model(train.x)), train.y)
    log(f'[epoch {e+1}] train loss: {loss}, train score: {tscore}, val score: {vscore}\n')
