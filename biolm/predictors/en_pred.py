import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from scipy.stats import spearmanr, pearsonr
from torch import nn
import pandas as pd
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import EsmTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from biolm.predictors.utils import get_tokenizer#, data_dir
from biolm.predictors.utils import _configure_optimizers
from transformers.models.esm.modeling_esm import *


class EnhancerDataset(Dataset):
    def __init__(self, df, mode='train', fold=None, tok_mode = 'char'):
        self.df = df
        self.mode = mode
        self.tok_mode = tok_mode
        self.tokenizer = get_tokenizer(tok_mode= tok_mode)
        self.fold = fold

        if self.fold is not None and self.mode == 'train':
            self.df = self.df[self.df['fold'] != self.fold].reset_index(drop=True)
        elif self.fold is not None and self.mode == 'valid':
            self.df = self.df[self.df['fold'] == self.fold].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)
        print(f'{mode} fold{self.fold} dataset size: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        seq = row['seq']
        activity = row['activity']
        activity = torch.FloatTensor([float(activity)])

        input_text = ''.join(seq)

        if self.tok_mode == 'bpe':
            pad_len = 64
            tokens = self.tokenizer.tokenize(input_text)
            features = {'input_ids': torch.from_numpy(np.asarray(tokens)).reshape((1, -1))}
            if features['input_ids'].shape[1] < pad_len:
                padding = torch.ones((1, pad_len - features['input_ids'].shape[1])) * self.tokenizer.pad_token_id
                features['input_ids'] = torch.cat([features['input_ids'], padding], dim=1).long()
            features['attention_mask'] = (features['input_ids'] != self.tokenizer.pad_token_id).long()

        elif self.tok_mode == '3mer':
            pad_len = 85
            # if tok mode is '3mer', then pad to 85
            features = self.tokenizer(input_text, return_tensors='pt')
            features['input_ids'] = F.pad(features['input_ids'], (0, pad_len - features['input_ids'].shape[1]), 'constant', self.tokenizer.pad_token_id)
            features['attention_mask'] = F.pad(features['attention_mask'], (0, pad_len - features['attention_mask'].shape[1]), 'constant', 0)
        else:
            features = self.tokenizer(input_text, return_tensors='pt')

        data_dict = {
                    "input_ids": features['input_ids'],
                    "attention_mask": features['attention_mask'],
                    "label": activity,
        }
        return data_dict

def get_data_loader(data_dir, ddp, batch_size, num_workers, species = 'mel', tok_mode = 'char', **kwargs):

    fold = 0

    if species == 'rice':
        random_df = pd.read_csv(f'{data_dir}/Enhancer_Activity_Prediction/E12.rice_Random.csv')
        en_df = pd.read_csv(f'{data_dir}/Enhancer_Activity_Prediction/E11.rice_Enhancer.csv')

    elif species == 'Arabidopsis':
        random_df = pd.read_csv(f'{data_dir}/Enhancer_Activity_Prediction/D22.Arabidopsis_Random_412.csv')
        en_df = pd.read_csv(f'{data_dir}/Enhancer_Activity_Prediction/D21.Arabidopsis_Enhancer_412.csv')

    elif species == 'mel':
        df_train = pd.read_csv(f'{data_dir}/Enhancer_Activity_Prediction/train.csv')
        df_train = df_train[df_train['fold'] != fold].reset_index(drop=True)
        ## test as val
        df_test = pd.read_csv(f'{data_dir}/Enhancer_Activity_Prediction/test.csv')
        df_test['fold'] = fold
        df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
        ## only for Dev label
        df.rename(columns={'Dev': 'activity'}, inplace=True)
        df = df[['seq', 'fold', 'activity']]

    if species != 'mel':
        df = pd.concat([random_df, en_df], axis=0).reset_index(drop=True)

    train_dataset = EnhancerDataset(df, mode='train', fold=fold, tok_mode = tok_mode)
    valid_dataset = EnhancerDataset(df, mode='valid', fold=fold, tok_mode = tok_mode)

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
        self.fc = nn.Linear(feat_num, 1)

        self.critertion = nn.SmoothL1Loss()
        self.last_loss = None

    def forward(self, input_ids, attn_mask, label=None):

        input_ids = input_ids.squeeze(1)
        attn_mask = attn_mask.squeeze(1)

        _, output = self.extractor(input_ids, attn_mask=attn_mask)
        pool_feat = F.adaptive_avg_pool1d(output.transpose(1, 2), 1).squeeze(2)
        output = self.dropout(pool_feat)
        output = self.fc(output)

        if label is not None:
            self.last_loss = self.critertion(output.view(-1), label.view(-1))

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
        X, attn_mask, Y = X.to(device), attn_mask.to(device), Y.to(device)

        with ctx:
            output = model(X, attn_mask, Y)
            losses.append(raw_model.last_loss.data.cpu().numpy())
            preds.extend(output.float().view(-1).detach().cpu().numpy().tolist())
            targets.extend(data_dict['label'].view(-1).detach().cpu().numpy().tolist())

    log_dict[f'{split}/loss'] = np.mean(losses)
    log_dict[f'{split}/PCC'] = pearsonr(preds, targets)[0]
    log_dict[f'{split}/SCC'] = spearmanr(preds, targets)[0]

    model.train()
    return log_dict

if __name__ == '__main__':
    # from biolm.llama2 import *
    # print("Loading a pretrained model")
    # ckpt_path = ('/public/home/taoshen/code/Projects/Zero2Hero/llama2/out_ckpts/mars/archived/'
    #              'mars_run-decoder-mars-s-add_attn_mask-2023_08_21_22_45_36/ckpt.pt')
    # extractor = load_model(ckpt_path, device='cpu')
    # predictor = Predictor(extractor)
    # print(predictor)
    # input_tensor = torch.randint(0, 21, (1, 128))
    # mask_tensor = torch.ones((1, 128))
    # label_tensor = None
    # output = predictor(input_tensor, mask_tensor, label_tensor)
    # print(output.shape)
    # print(predictor.last_loss)

    data_dir = '/public_new/taoshen/data/mars_fm_data/downstream'
    ddp, batch_size, num_workers = False, 32, 0
    train_sampler, train_loader, valid_loader, _ = get_data_loader(data_dir, ddp, batch_size, num_workers, tok_mode='MarsTok')

    for data_dict in train_loader:
        print(data_dict['input_ids'].shape)
