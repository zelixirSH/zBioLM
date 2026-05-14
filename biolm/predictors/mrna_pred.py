import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from biolm.predictors.utils import get_tokenizer#, data_dir
from biolm.predictors.utils import _configure_optimizers
from transformers.models.esm.modeling_esm import *

from sklearn.metrics import mean_squared_error

def mcrmse_loss_fn(outputs, targets):
    colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss

def my_mcrmse(outputs, targets):
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    all_rmse = []

    for i in range(5):
        rmse = np.sqrt(np.mean(mean_squared_error(targets[:, :, i], outputs[:, :, i]), axis=1))
        all_rmse.append(rmse)
    mcrmse = np.mean(all_rmse)
    return mcrmse

class MRNADataset(Dataset):
    def __init__(self, df, mode='train', fold=None):
        self.df = df
        self.mode = mode
        self.tokenizer = get_tokenizer()
        self.fold = fold

        if self.fold is not None and self.mode == 'train':
            self.df = self.df[self.df['fold'] != self.fold].reset_index(drop=True)
        elif self.fold is not None and self.mode == 'valid':
            self.df = self.df[self.df['fold'] == self.fold].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)
        print(f'{mode} fold{self.fold} dataset size: {len(self.df)}')

        self.pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        seq = row['sequence'].replace('U', 'T')

        if self.mode == 'train' or self.mode == 'valid':
            labels = np.array(row[self.pred_cols].values.tolist()).transpose((1, 0))
            labels = torch.FloatTensor(labels)

        input_text = ' '.join(seq)
        features = self.tokenizer(input_text, return_tensors='pt')

        if self.mode == 'train' or self.mode == 'valid':
            data_dict = {"input_ids": features['input_ids'],
                         "label": labels,
                         "attention_mask": features['attention_mask']}
        else:
            data_dict = {
                        "input_ids": features['input_ids'],
                         "attention_mask": features['attention_mask']}

        return data_dict


def get_data_loader(data_dir, ddp, batch_size, num_workers, **kwargs):


    fold = 0

    df = pd.read_csv(f'{data_dir}/mRNA_degradation_prediction/kaggle_train.csv')

    recover_cols = ['reactivity_error',
                    'deg_error_Mg_pH10', 'deg_error_pH10',
                    'deg_error_Mg_50C', 'deg_error_50C',
                    'reactivity', 'deg_Mg_pH10',
                    'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    def fun_str_list(x):
        x = eval(x)
        return x

    for col in recover_cols:
        df[col] = df[col].apply(fun_str_list)

    df = df[df.signal_to_noise > 1].reset_index(drop=True)

    train_dataset = MRNADataset(df, mode='train', fold=fold)
    valid_dataset = MRNADataset(df, mode='valid', fold=fold)

    # # DistributedSampler will partition the training dataset into num_replicas
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not ddp), num_workers=num_workers, sampler=train_sampler)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_sampler, train_loader, valid_loader, None

class Predictor(nn.Module):
    def __init__(self, extractor, dropout=0.1, is_freeze=False, **kwargs):
        super(Predictor, self).__init__()

        self.freeze = is_freeze
        self.extractor = extractor
        feat_num = extractor.params.dim

        if self.freeze:
            for param in self.extractor.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_num, 5)

        self.critertion =  mcrmse_loss_fn
        self.last_loss = None
        
    def forward(self, input_ids, attn_mask, label=None):

        # HOTFIX
        pred_len = 68
        input_ids = input_ids.squeeze(1)
        attn_mask = attn_mask.squeeze(1)

        _, output = self.extractor(input_ids, attn_mask=attn_mask)

        feat = output  ## [bs, seq_len, feat_num]
        output = self.dropout(feat)
        output = self.fc(output)

        if pred_len is not None:
            output = output[:, 1:pred_len + 1, :]

        if label is not None:
            label = label.squeeze(1)
            self.last_loss = self.critertion(output, label).mean()

        return output

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        return _configure_optimizers(self.named_parameters(), weight_decay, learning_rate, betas, device_type)



@torch.no_grad()
def estimate_metrics(ctx, model, raw_model, split, data_loader, device):
    log_dict = {}

    if data_loader is None:
        return log_dict

    preds, targets = [], []
    model.eval()

    losses = []
    for data_dict in tqdm(data_loader):
        X, attn_mask, Y = data_dict["input_ids"], data_dict["attention_mask"], data_dict["label"]
        X, attn_mask, Y  = X.to(device), attn_mask.to(device), Y.to(device)

        with ctx:
            output = model(X, attn_mask, Y)
            losses.append(raw_model.last_loss.data.cpu().numpy())

            preds.append(output.detach())
            targets.append(Y.detach())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    mcrmse = mcrmse_loss_fn(preds, targets).mean().item()

    log_dict[f'{split}/loss'] = np.mean(losses)
    log_dict[f'{split}/MCRMSE'] = mcrmse

    model.train()
    return log_dict


if __name__ == '__main__':
    from biolm.llama2 import *
    print("Loading a pretrained model")
    ckpt_path = 'ckpt.pt'
    extractor = load_model(ckpt_path, device='cpu')
    predictor = Predictor(extractor)
    print(predictor)
    input_tensor = torch.randint(0, 21, (1, 128))
    mask_tensor = torch.ones((1, 128))
    label_tensor = None
    output = predictor(input_tensor, mask_tensor, label_tensor)
    print(output.shape)
    print(predictor.last_loss)