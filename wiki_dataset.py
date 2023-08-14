"""
NoisyWikihow DataSet & Models

Note that some models only adapt to Bart, since different pretrained models differ in many aspects,
e.g: config, pad_token_id, initialization strategies, classification heads.

Any Model can run 'base' method correctly.
Bart can run every methods.
"""


import os
import sys
sys.path.insert(0,'..')
from torch.utils.data import DataLoader, Dataset
# from cmd_args import args
import torch.nn as nn

import pandas as pd
import numpy as np
from transformers import BartConfig, BartTokenizer, BartForSequenceClassification
from torch.utils.data import Dataset

MODEL_NAME = "facebook/bart-base"

'''
Data Format: x,y,index
Noise on feature: train(x_noisy, y) test(x,y) e.g: mix, tail, uncommon, neighbor
Noise on label: train(x, y_noisy) test(x,y)   e.g: sym, idn
'''
class WikiDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.train_labels = np.array(y)
    # x, y, index
    def __getitem__(self, index):
        item = {k:v[index] for k,v in self.x.items()} 
        return item, self.train_labels[index], index
    def __len__(self):
        return len(self.train_labels)
    def update_corrupted_label(self, noise_label):
        self.train_labels[:] = noise_label[:]

tokenizer = None
def get_wiki_tokenizer(args):
    global tokenizer
    if not tokenizer:
        print(f"Loading the tokenizer for bart")
        # We have changed steps and intention label into lower cased.
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
        print("tokenizer prepared")
    return tokenizer


def get_wiki_train_and_val_loader(args):
    tokenizer = get_wiki_tokenizer(args)
    print('==> Preparing data for sst..')
    val_csv = pd.read_csv(f"{args.data_path}/noisy/val.csv")
    val_step = tokenizer(val_csv["step"].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    val_cat = val_csv["cat_id"].to_list()

    test_csv = pd.read_csv(f"{args.data_path}/noisy/test.csv")
    test_step = tokenizer(test_csv["step"].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    test_cat = test_csv["cat_id"].to_list()

    if args.noise_rate!=0:
        train_csv = pd.read_csv(f"{args.data_path}/noisy/{args.noise_mode}_{args.noise_rate:.1f}.csv")
        train_step = train_csv["step_id"].to_list()
        train_cat = train_csv["cat_id"].to_list()
        train_num = len(train_cat) 
        # Noise on feature:  train(x_noisy, y)
        if 'noisy_id' in train_csv:
            train_noisy = train_csv["noisy_id"].to_list()
            noisy_ind = [i for i in range(train_num) if train_step[i]!=train_noisy[i]]
            clean_ind = [i for i in range(train_num) if train_step[i]==train_noisy[i]]
            train_noisy = tokenizer(train_csv["noisy_step"].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            trainset = WikiDataSet(train_noisy, train_cat)

        # Noise on label:  train(x, y_noisy)
        else:
            noisy_y = train_csv["noisy_label"].to_list()
            noisy_ind = [i for i in range(train_num) if train_cat[i]!=noisy_y[i]]
            clean_ind = [i for i in range(train_num) if train_cat[i]==noisy_y[i]]
            train_step = tokenizer(train_csv["step"].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            trainset = WikiDataSet(train_step, noisy_y)

    else:
        train_csv = pd.read_csv(f"{args.data_path}/noisy/train.csv")
        train_step = tokenizer(train_csv["step"].to_list(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        train_cat = train_csv["cat_id"].to_list()
        train_num = len(train_cat)
        noisy_ind = []
        clean_ind = list(range(train_num))
        trainset = WikiDataSet(train_step, train_cat)

    valset = WikiDataSet(val_step, val_cat)
    testset = WikiDataSet(test_step, test_cat)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    save_path = f'data_meta/wiki/{args.noise_rate}'
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(f"{save_path}/noisy_ind.npy"):
        old_noisy_ind = np.load(f"{save_path}/noisy_ind.npy")
        assert np.all(old_noisy_ind == np.array(noisy_ind))
    np.save(f"{save_path}/noisy_ind.npy", noisy_ind)
    np.save(f"{save_path}/clean_ind.npy", clean_ind)
    
    return train_loader, val_loader, test_loader, noisy_ind, clean_ind

from PreResNet_rours import ULTRA
class CustomClassificationHead(nn.Module):
    def __init__(self, classification_head, func, filter=None):
        super(CustomClassificationHead, self).__init__()
        self.func = func
        self.filter = filter
        self.classification_head = classification_head
    def forward(self, hidden_states):
        # print('hidden_state', hidden_states.shape)
        hidden_states = self.func(hidden_states, filter=self.filter)
        hidden_states = self.classification_head(hidden_states)
        return hidden_states

def get_wiki_model(args, num_class = 158):
    config = BartConfig.from_pretrained(MODEL_NAME, num_labels = num_class)
    ##############
    print('Loading {}'.format(MODEL_NAME))
    model = BartForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    func = ULTRA(args)
    model.classification_head = CustomClassificationHead(model.classification_head, func)
    print('Custom CLSHead for {}'.format(MODEL_NAME))
    return model

# def get_wiki_tokenizer_and_label(args):
#     print('Prepare tokenizer and label for {}'.format(args.model_type))
#     tokenizer = get_wiki_tokenizer(args)
#     # Do lower case for labels, make it easier to match up
#     cat = pd.read_csv(f'{args.data_path}/cat158.csv')['category'].map(lambda x:x.lower()).to_list()
#     cat_token = tokenizer(cat, padding='max_length', max_length=15, truncation=True, return_tensors='pt')['input_ids'].to(args.device)
#     cat_labels = tokenizer.batch_decode(cat_token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     cat_token[cat_token==tokenizer.pad_token_id]=-100
#     return tokenizer, cat_token, cat_labels

# def save_index():
#     import os
#     import json
#     index_path = 'corrupt_index'
#     os.makedirs(index_path,exist_ok=True)
#     for mode in ['mix','sym','idn']:
#         args.noise_mode = mode
#         print(mode)
#         home = index_path
#         for nr in [0.1, 0.2, 0.4, 0.6]:
#             args.noise_rate = nr
#             train_loader, test_loader, noisy_ind, clean_ind = get_wiki_train_and_val_loader(args)
#             with open(os.path.join(home,f'{mode}_{nr}_noisy.txt'),"w") as f:
#                 json.dump(noisy_ind, f)
#             print(f"nr{nr}: noisy{len(noisy_ind)}\tclean{len(clean_ind)}")


if __name__ == '__main__':
    pass