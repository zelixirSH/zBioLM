import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
from torch import nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from biolm.predictors.utils import _configure_optimizers
from transformers.models.esm.modeling_esm import *
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from biolm.predictors.utils import get_tokenizer

class DynamicPaddingCollateFn:
    def __init__(self, tok_mode = 'char'):
        self.tok_mode = tok_mode
        self.tokenizer = get_tokenizer(tok_mode)

    def __call__(self, batch):
        seqs, labels = zip(*batch)

        if self.tok_mode == '3mer':
            max_len = max([len(seq)//3 + 2 for seq in seqs])
        else:
            max_len = max([len(seq) + 2 for seq in seqs])

        if self.tok_mode == 'bpe':
            pad_len = 256+128

            feat_all = []
            for input_text in seqs:
                tokens = self.tokenizer.tokenize(input_text)
                features = {'input_ids': torch.from_numpy(np.asarray(tokens)).reshape((1, -1))}

                if features['input_ids'].shape[1] < pad_len:
                    padding = torch.ones((1, pad_len - features['input_ids'].shape[1])) * self.tokenizer.pad_token_id
                    features['input_ids'] = torch.cat([features['input_ids'], padding], dim=1).long()
                features['attention_mask'] = (features['input_ids'] != self.tokenizer.pad_token_id).long()

                feat_all.append(features)

            feat = {}
            feat['input_ids'] = torch.cat([f['input_ids'] for f in feat_all], dim=0)
            feat['attention_mask'] = torch.cat([f['attention_mask'] for f in feat_all], dim=0)

        else:
            feat = self.tokenizer.batch_encode_plus(seqs,
                                                    padding='max_length',
                                                    max_length=max_len,
                                                    truncation=True,
                                                    return_tensors='pt')


        data_dict = {"input_ids": feat['input_ids'],
                     "attention_mask": feat['attention_mask']}
        data_dict['label'] = torch.FloatTensor(labels)

        return data_dict

class PADataset(Dataset):
    def __init__(self, df, mode='train', fold=None, crop_size=None, tok_mode = 'char'):
        self.df = df
        self.mode = mode
        self.tokenizer = get_tokenizer(tok_mode=tok_mode)
        self.fold = fold
        self.crop_size = crop_size

        if self.fold is not None and self.mode == 'train':
            self.df = self.df[self.df['fold_c4'] != self.fold].reset_index(drop=True)
        elif self.fold is not None and self.mode == 'valid':
            self.df = self.df[self.df['fold_c4'] == self.fold].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)
        print(f'{mode} fold{self.fold} dataset size: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        seq = row['dna_seqs'].upper().replace('U', 'T')

        if self.crop_size and len(seq) > self.crop_size:
            start_idx = np.random.randint(len(seq) - self.crop_size + 1, size=1)[0]
            seq = seq[start_idx: start_idx + self.crop_size]

        label = float(row['abundance_log_value'])

        return seq, label


def get_data_loader(data_dir, ddp, batch_size, num_workers, tok_mode='char', **kwargs):


    fold = 0
    crop_size = 512

    df = pd.read_csv(f'{data_dir}/Protein_Abundance_Prediction/ecoli_protein_gene_abundance_dataset.csv')
    dynamic_padding_collate_fn = DynamicPaddingCollateFn(tok_mode)
    train_dataset = PADataset(df, mode='train', fold=fold, crop_size=crop_size * 3, tok_mode=tok_mode)

    # DistributedSampler will partition the training dataset into num_replicas
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not ddp), num_workers=num_workers,
                              collate_fn=dynamic_padding_collate_fn, sampler=train_sampler)

    valid_dataset = PADataset(df, mode='valid', fold=fold, crop_size=crop_size * 3, tok_mode=tok_mode)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                              collate_fn=dynamic_padding_collate_fn)

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
        X, attn_mask, Y  = X.to(device), attn_mask.to(device), Y.to(device)

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
    ddp, batch_size, num_workers = False, 32, 0

    data_dir = '/public_new/taoshen/data/mars_fm_data/downstream'
    train_sampler, train_loader, valid_loader, _ = get_data_loader(data_dir, ddp, batch_size, num_workers, tok_mode='MarsTok')
    for data_dict in train_loader:
        print(data_dict['input_ids'].shape)
