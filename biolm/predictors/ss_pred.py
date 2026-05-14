
import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import EsmTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

from biolm.modules.resnet import resnet_b16, resnet_b1
from biolm.predictors.utils import _configure_optimizers
from transformers.models.esm.modeling_esm import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import torch
from torch import nn
from biolm.predictors.utils import get_tokenizer

tokenizer = None

def collate_fn(batch):
    '''

    '''
    seqs, cts = zip(*batch)
    max_len = max([len(seq)+2 for seq in seqs])

    if isinstance(tokenizer, EsmTokenizer):
        data_dict = tokenizer.batch_encode_plus(seqs,
                                                padding='max_length',
                                                max_length=max_len,
                                                truncation=True,
                                                return_tensors='pt')
    else:
        # pad to max_len
        seqs = ["<bos>" + seq + "<eos>" + '<pad>' * (max_len - 2 - len(seq)) for seq in seqs]

        tok_ids = []
        for seq in seqs:
            tok_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq, mode='char'))
            tok_ids.append(tok_id)

        tok_ids = np.array(tok_ids)
        data_dict = {
            'input_ids': torch.tensor(tok_ids, dtype=torch.int),
            'attention_mask': torch.tensor(tok_ids != tokenizer._convert_token_to_id('<pad>'), dtype=torch.int),
        }

    ## padding ct
    ct_masks = [np.ones(ct.shape) for ct in cts]
    cts = [np.pad(ct, (0, max_len-ct.shape[0]), 'constant') for ct in cts]
    ## padding ct_mask
    ct_masks = [np.pad(ct_mask, (0, max_len-ct_mask.shape[0]), 'constant') for ct_mask in ct_masks]
    data_dict['label'] = torch.FloatTensor(cts)
    data_dict['label_mask'] = torch.FloatTensor(ct_masks)

    return data_dict


class SSDataset(Dataset):
    def __init__(self, df, data_path, tok_mode = 'char'):
        self.df = df
        self.data_path = data_path
        global tokenizer
        tokenizer = get_tokenizer(tok_mode)
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        seq = seq.replace('U', 'T')
        file_name = row['file_name']
        file_path = os.path.join(self.data_path, file_name + '.npy')
        ct = np.load(file_path)
        return seq, ct


def get_data_loader(data_dir, ddp, batch_size, num_workers, tok_mode = 'char', db = 'bprna', **kwargs):
    ## bprna data ##

    if db == 'bprna':
        bprna_dir = f'{data_dir}/RNA_Secondary_Structure_Prediction/bpRNA'
        df = pd.read_csv(f'{bprna_dir}/bpRNA.csv')

        df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)
        train_dataset = SSDataset(df_train, data_path=f'{bprna_dir}/ct/TR0', tok_mode=tok_mode)

        # DistributedSampler will partition the training dataset into num_replicas
        train_sampler = DistributedSampler(train_dataset) if ddp else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not ddp), num_workers=num_workers,
                                  collate_fn=collate_fn, sampler=train_sampler)

        df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)
        valid_dataset = SSDataset(df_val, data_path=f'{bprna_dir}/ct/VL0', tok_mode=tok_mode)
        test_dataset = SSDataset(df_test, data_path=f'{bprna_dir}/ct/TS0', tok_mode=tok_mode)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn)

    elif db == 'pdb_hw':
        bprna_dir = f'{data_dir}/RNA_Secondary_Structure_Prediction/PDB_hw'
        df = pd.read_csv(f'{bprna_dir}/dssr_unmodel_nc_new_updated.csv')

        df_train = df[df['data_name'] == 'TR1'].reset_index(drop=True)
        train_dataset = SSDataset(df_train, data_path=f'{bprna_dir}/ct_new_unmodel', tok_mode=tok_mode)

        # DistributedSampler will partition the training dataset into num_replicas
        train_sampler = DistributedSampler(train_dataset) if ddp else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not ddp), num_workers=num_workers,
                                  collate_fn=collate_fn, sampler=train_sampler)

        df_val = df[df['data_name'] == 'VL1'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS1'].reset_index(drop=True)
        valid_dataset = SSDataset(df_val, data_path=f'{bprna_dir}/ct_new_unmodel', tok_mode=tok_mode)
        test_dataset = SSDataset(df_test, data_path=f'{bprna_dir}/ct_new_unmodel', tok_mode=tok_mode)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn)

    else:
        raise NotImplementedError

    return train_sampler, train_loader, valid_loader, test_loader


class SSCNNPredictor(nn.Module):
    def __init__(self, extractor, is_freeze = False, mode = 'cnn16', **kwargs):
        super(SSCNNPredictor, self).__init__()
        self.extractor = extractor

        self.feat_num = extractor.params.dim
        if mode == 'cnn1':
            self.cnn = resnet_b1(myChannels=self.feat_num)
        elif mode == 'cnn16':
            self.cnn = resnet_b16(myChannels=self.feat_num, bbn=16)
        else:
            raise NotImplementedError

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

        self.criterion = nn.BCEWithLogitsLoss()
        self.last_loss = None

    def forward_(self, data_dict):

        input_ids, attn_mask = data_dict['input_ids'], data_dict['attention_mask']

        if self.is_freeze:
            with torch.no_grad():
                _, output = self.extractor(input_ids, attn_mask=attn_mask)
        else:
            _, output = self.extractor(input_ids, attn_mask=attn_mask)

        ## L*ch-> LxL*ch
        matrix = torch.einsum('ijk,ilk->ijlk', output, output)
        matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d
        x = self.cnn(matrix)
        x = x.squeeze(-1)

        return x

    def forward(self, input_ids, attn_mask, label=None):

        if self.is_freeze:
            with torch.no_grad():
                _, output = self.extractor(input_ids, attn_mask=attn_mask)
        else:
            _, output = self.extractor(input_ids, attn_mask=attn_mask)

        ## L*ch-> LxL*ch
        matrix = torch.einsum('ijk,ilk->ijlk', output, output)
        matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d
        x = self.cnn(matrix)
        x = x.squeeze(-1)

        if label is not None:
            logits = x
            labels = label
            loss_list = []
            bs = logits.shape[0]

            for idx in range(bs):
                ## exclude padding ##
                seq_length = attn_mask[idx].sum().item()
                logit = logits[idx, :seq_length, :seq_length]
                ## exclude padding ##
                ## exclude start and end token ##
                logit = logit[1:-1, 1:-1]
                label = labels[idx, :logit.shape[0], :logit.shape[1]]
                ## exclude start and end token ##
                loss_list.append(self.criterion(logit.contiguous().view(-1), label.contiguous().view(-1)))

            self.last_loss  = torch.stack(loss_list).mean()

        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        return _configure_optimizers(self.named_parameters(), weight_decay, learning_rate, betas, device_type)


@torch.no_grad()
def estimate_metrics(ctx, model, raw_model, split, data_loader, device):
    log_dict = {}

    threshold = 0.5
    val_auc_list, val_recall_list, val_precision_list, val_f1_list = [], [], [], []

    model.eval()

    losses = []
    for data_dict in tqdm(data_loader):
        X, attn_mask, Y = data_dict["input_ids"], data_dict["attention_mask"], data_dict["label"]
        X, attn_mask, Y  = X.to(device), attn_mask.to(device), Y.to(device)

        with ctx:
            logits = model(X, attn_mask, Y)
            losses.append(raw_model.last_loss.data.cpu().numpy())
            bs = logits.shape[0]

            for idx in range(bs):
                seq_length = attn_mask[idx].sum().item()
                logit = logits[idx, :seq_length, :seq_length]
                logit = logit[1:-1, 1:-1]
                label = Y[idx, :logit.shape[0], :logit.shape[1]]

                probs = torch.sigmoid(logit.float()).detach().cpu().numpy()
                pred = (probs > threshold).astype(np.float32)
                val_recall_list.append(recall_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                val_precision_list.append(
                    precision_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                val_f1_list.append(f1_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))

    log_dict[f'{split}/loss'] = np.mean(losses)
    log_dict[f'{split}/auc'] = np.mean(val_auc_list)
    log_dict[f'{split}/recall'] = np.mean(val_recall_list)
    log_dict[f'{split}/precision'] = np.mean(val_precision_list)
    log_dict[f'{split}/f1'] = np.mean(val_f1_list)

    model.train()
    return log_dict


def unitest_model():
    print("Loading a pretrained model")
    ckpt_path = 'ckpt.pt'
    extractor = load_model(ckpt_path, device='cpu')
    predictor = SSCNNPredictor(extractor)
    print(predictor)
    input_tensor = torch.randint(0, 21, (1, 128))
    mask_tensor = torch.ones((1, 128))
    label_tensor = None
    output = predictor(input_tensor, mask_tensor, label_tensor)
    print(output.shape)
    print(predictor.last_loss)


def unitest_data():
    '''

    '''
    # seed_everything(42)

    ddp = False
    batch_size = 2
    num_workers = 0
    tok_mode = 'MarsTok'
    data_dir = '/public_new/taoshen/data/mars_fm_data/downstream'
    train_sampler, train_loader, valid_loader, test_loader = \
        get_data_loader(data_dir, ddp, batch_size, num_workers, tok_mode)
    print(len(train_loader))


if __name__ == '__main__':
    from biolm.llama2 import *
    # unitest_model()
    unitest_data()