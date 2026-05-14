from scipy.stats import spearmanr, pearsonr
from torch import nn
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from biolm.predictors.utils import _configure_optimizers
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, matthews_corrcoef
from transformers.models.esm.modeling_esm import *
from torch.utils.data import Dataset
from biolm.predictors.utils import get_tokenizer#, data_dir
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class CLSDataset(Dataset):
    def __init__(self, df, mode='train', fold=None, tok_mode='MarsTok'):
        self.df = df
        self.mode = mode
        self.tok_mode = tok_mode
        self.tokenizer = get_tokenizer(tok_mode = tok_mode)
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
        seq = row['seq'].upper().replace('U', 'T')
        label = row['label']
        label = torch.FloatTensor([float(label)])
        # BUG fix here ' '.join -> ''.join
        input_text = ''.join(seq)

        if self.tok_mode == 'bpe':
            pad_len = 64
            tokens = self.tokenizer.tokenize(input_text)
            features = {'input_ids': torch.from_numpy(np.asarray(tokens)).reshape((1, -1))}

            if features['input_ids'].shape[1] < pad_len:
                padding = torch.ones((1, pad_len - features['input_ids'].shape[1])) * self.tokenizer.pad_token_id
                features['input_ids'] = torch.cat([features['input_ids'], padding], dim=1).long()
            features['attention_mask'] = (features['input_ids'] != self.tokenizer.pad_token_id).long()
        else:
            features = self.tokenizer(input_text, return_tensors='pt')

        data_dict = {"input_ids": features['input_ids'],
                     "label": label,
                     "attention_mask": features['attention_mask'],}

        return data_dict

def get_data_loader(data_dir, ddp, batch_size, num_workers, cls_task = 'promoter', tok_mode='MarsTok', **kwargs):

    fold = 0

    df = pd.read_csv(f'{data_dir}/cls_task/{cls_task}/train.csv')
    train_dataset = CLSDataset(df, mode='train', fold=fold, tok_mode = tok_mode)
    valid_dataset = CLSDataset(df, mode='valid', fold=fold, tok_mode = tok_mode)

    # # DistributedSampler will partition the training dataset into num_replicas
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not ddp), num_workers=num_workers, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_sampler, train_loader, valid_loader, None

#########################################################################################################################
class sirnaDataset(Dataset):
    def __init__(self, df, mode='train', fold=None, crop_size=None, tok_mode='MarsTok'):
        self.df = df
        self.mode = mode
        self.tokenizer = get_tokenizer(tok_mode=tok_mode)
        self.fold = fold
        self.crop_size = crop_size
        self.df = self.df.reset_index(drop=True)
        print(f'{mode} fold{self.fold} dataset size: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        seq_0 = row['Antisense(21 mer)'].upper().replace('U', 'T')
        seq_1 = row['Sense(19 mer)'].upper().replace('U', 'T')
        seq = seq_0 +'<unk>'+ seq_1

        if self.crop_size and len(seq) > self.crop_size:
            start_idx = np.random.randint(len(seq) - self.crop_size + 1, size=1)[0]
            seq = seq[start_idx: start_idx + self.crop_size]

        label = float(row['%Inhibition']) / 100.0

        # label = torch.FloatTensor([float(label)])
        if label > 0.7:
            label = 1
        else:
            label = 0

        return seq, label

class DynamicPaddingCollateFn:
    def __init__(self, tok_mode = 'MarsTok'):
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

            data_dict = {"input_ids": feat['input_ids'],
                         "attention_mask": feat['attention_mask']}
            data_dict['label'] = torch.FloatTensor(labels)

        elif self.tok_mode == 'char':
            feat = self.tokenizer.batch_encode_plus(seqs,
                                                    padding='max_length',
                                                    max_length=max_len,
                                                    truncation=True,
                                                    return_tensors='pt')
            data_dict = {"input_ids": feat['input_ids'],
                         "attention_mask": feat['attention_mask']}
            data_dict['label'] = torch.FloatTensor(labels)

        else:
            # pad to max_len
            seqs = ["<bos>" + seq + "<eos>" + '<pad>' * (max_len - 2 - len(seq)) for seq in seqs]

            tok_ids = []
            for seq in seqs:
                tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq, mode='char'))
                tok_ids.append(tok_id)

            tok_ids = np.array(tok_ids)
            data_dict = {
                'input_ids': torch.tensor(tok_ids, dtype=torch.int),
                'attention_mask': torch.tensor(tok_ids != self.tokenizer._convert_token_to_id('<pad>'), dtype=torch.int),
            }
            data_dict['label'] = torch.FloatTensor(labels)

        return data_dict

def get_data_loader_sirna(data_dir, ddp, batch_size, num_workers, tok_mode='MarsTok', **kwargs):
    '''

    '''

    fold = 0
    crop_size = 64

    dynamic_padding_collate_fn = DynamicPaddingCollateFn(tok_mode)

    df = pd.read_csv(f'{data_dir}/siRNA/siRNA_TR2182.csv')
    train_dataset_0 = sirnaDataset(df, mode='train', fold=fold, crop_size=crop_size, tok_mode=tok_mode)

    df = pd.read_csv(f'{data_dir}/siRNA/agt_50_percent.csv')
    train_dataset_1 = sirnaDataset(df, mode='train', fold=fold, crop_size=crop_size, tok_mode=tok_mode)
    train_dataset = train_dataset_0 + train_dataset_1

    df = pd.read_csv(f'{data_dir}/siRNA/siRNA_TE249.csv')
    valid_dataset = sirnaDataset(df, mode='valid', fold=fold, crop_size=crop_size, tok_mode=tok_mode)
    train_dataset += valid_dataset

    # DistributedSampler will partition the training dataset into num_replicas
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not ddp), num_workers=num_workers,
                              collate_fn=dynamic_padding_collate_fn, sampler=train_sampler)

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                              collate_fn=dynamic_padding_collate_fn)

    df = pd.read_csv(f'{data_dir}/siRNA/agt_50_percent_1.csv') #agt_50_percent_1.csv
    test_dataset = sirnaDataset(df, mode='test', fold=fold, crop_size=crop_size, tok_mode=tok_mode)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                             collate_fn=dynamic_padding_collate_fn)

    return train_sampler, train_loader, valid_loader, test_loader

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

        self.criterion = nn.BCEWithLogitsLoss()
        self.last_loss = None

    def forward(self, input_ids, attn_mask, label=None):

        input_ids = input_ids.squeeze(1)
        attn_mask = attn_mask.squeeze(1)

        _, output = self.extractor(input_ids, attn_mask=attn_mask)
        pool_feat = F.adaptive_avg_pool1d(output.transpose(1, 2), 1).squeeze(2)
        output = self.dropout(pool_feat)
        output = self.fc(output)

        if label is not None:
            # loss = criterion(output.view(-1), data_dict['label'].view(-1))
            self.last_loss = self.criterion(output.view(-1), label.view(-1))

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

    threshold = 0.5
    acc = accuracy_score(np.array(preds) > threshold, targets)
    auc = roc_auc_score(targets, preds)
    mcc = matthews_corrcoef(np.array(preds) > threshold, targets)
    f1 = f1_score(np.array(preds) > threshold, targets)
    recall = recall_score(np.array(preds) > threshold, targets)

    log_dict[f'{split}/acc'] = acc
    log_dict[f'{split}/auc'] = auc
    log_dict[f'{split}/mcc'] = mcc
    log_dict[f'{split}/f1'] = f1
    log_dict[f'{split}/recall'] = recall

    # draw PR curve
    precision, recall, thresholds = precision_recall_curve(targets, preds)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # save the plot
    plt.savefig(f'{split}_PR_curve.png')
    # plt.show()
    plt.close()

    # draw roc curve
    fpr, tpr, thresholds = roc_curve(targets, preds)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # save the plot
    plt.savefig(f'{split}_ROC_curve.png')
    plt.close()

    model.train()
    return log_dict

if __name__ == '__main__':
    # from biolm.llama2 import *
    # print("Loading a pretrained model")
    # ckpt_path = '/public/home/taoshen/code/Projects/Zero2Hero/llama2/out_ckpts/mars/archived/mars_run-decoder-mars-s-add_attn_mask-2023_08_21_22_45_36/ckpt.pt'
    # extractor = load_model(ckpt_path, device='cpu')
    # predictor = Predictor(extractor)
    # print(predictor)
    # input_tensor = torch.randint(0, 21, (1, 128))
    # mask_tensor = torch.ones((1, 128))
    # label_tensor = None
    # output = predictor(input_tensor, mask_tensor, label_tensor)
    # print(output.shape)
    # print(predictor.last_loss)

    ddp, batch_size, num_workers = False, 32, 0

    # train_sampler, train_loader, valid_loader, _ = get_data_loader(ddp, batch_size, num_workers, cls_task='promoter', tok_mode='bpe')
    #
    # for data_dict in train_loader:
    #     print(data_dict['input_ids'].shape)
    #     print(data_dict['attention_mask'].shape)
    #     print(data_dict['label'].shape)
    #     break
    #
    # ddp, batch_size, num_workers = False, 32, 0
    # train_sampler, train_loader, valid_loader, _ = get_data_loader(ddp, batch_size, num_workers, cls_task='promoter', tok_mode='3mer')
    #
    # for data_dict in train_loader:
    #     print(data_dict['input_ids'].shape)
    #     print(data_dict['attention_mask'].shape)
    #     print(data_dict['label'].shape)
    #     break

    data_dir = '/public_new/taoshen/data/mars_fm_data/downstream'
    train_sampler, train_loader, valid_loader, _ = \
        get_data_loader_sirna(data_dir, ddp, batch_size, num_workers)

    for data_dict in train_loader:
        print(data_dict['input_ids'].shape)
        print(data_dict['attention_mask'].shape)
        print(data_dict['label'].shape)
        break
