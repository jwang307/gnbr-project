# coding=UTF-8
import os
import argparse
from tqdm import tqdm
import torch
# import pdb
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from trainer import Trainer
from dataloader import DataLoader
from model import LMKE

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--model_lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('--data', type=str, default='chemical-disease')
    parser.add_argument('--plm', type=str, default='biobert', choices=['bert', 'biobert'])
    parser.add_argument('--description', type=str, default='desc')

    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=-1)
    parser.add_argument('--load_metric', type=str, default='hits1')

    # directly run test
    parser.add_argument('--link_prediction', default=False, action='store_true')
    parser.add_argument('--triple_classification', default=False, action='store_true')

    parser.add_argument('--add_tokens', default=False, action='store_true',
                        help='add entity and relation tokens into the vocabulary')
    parser.add_argument('--p_tuning', default=False, action='store_true', help='add learnable soft prompts')
    parser.add_argument('--prefix_tuning', default=False, action='store_true',
                        help='fix language models and only tune added components')
    parser.add_argument('--rdrop', default=False, action='store_true')
    parser.add_argument('--self_adversarial', default=False, action='store_true',
                        help='self adversarial negative sampling')
    parser.add_argument('--no_use_lm', default=False, action='store_true')
    parser.add_argument('--use_structure', default=False, action='store_true')
    parser.add_argument('--contrastive', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--load_descriptions', default=False, action='store_true')

    parser.add_argument('--task', default='LP', choices=['LP', 'TC'])

    arg = parser.parse_args()

    if arg.task == 'TC':
        neg_rate = 1
    else:
        neg_rate = 0

    identifier = '{}-{}-{}-batch_size={}-prefix_tuning={}'.format(arg.data, arg.plm, arg.description, arg.batch_size,
                                                                  arg.prefix_tuning)

    # Set random seed
    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    try:
        device = torch.device('mps')
    except Exception as e:
        device = torch.device('cpu')
        print(e)

    if arg.plm == 'bert':
        plm_name = "bert-base-uncased"
        t_model = 'bert'
    elif arg.plm == 'biobert':
        plm_name = "dmis-lab/biobert-v1.1"
        t_model = 'bert'

    if arg.data == 'chemical-gene':
        in_paths = {
            'dataset': arg.data,
            'train': './data/FB13/train.tsv',
            'valid': './data/FB13/dev.tsv',
            'test': './data/FB13/test.tsv',
            'text': ['./data/FB13/entity2text.txt', './data/FB13/relation2text.txt']
        }

    lm_config = AutoConfig.from_pretrained(plm_name, cache_dir='./cached_model')
    lm_tokenizer = AutoTokenizer.from_pretrained(plm_name, do_basic_tokenize=False, cache_dir='./cached_model')
    lm_model = AutoModel.from_pretrained(plm_name, config=lm_config, cache_dir='./cached_model')

    data_loader = DataLoader(in_paths, lm_tokenizer, batch_size=arg.batch_size, neg_rate=neg_rate,
                             add_tokens=arg.add_tokens, p_tuning=arg.p_tuning, rdrop=arg.rdrop, model=t_model,
                             descriptions=arg.load_descriptions)

    if arg.add_tokens:
        data_loader.adding_tokens()
        lm_model.resize_token_embeddings(len(lm_tokenizer))

    model = LMKE(lm_model, n_ent=len(data_loader.ent2id), n_rel=len(data_loader.rel2id), add_tokens=arg.add_tokens,
                 contrastive=arg.contrastive)

    no_decay = ["bias", "LayerNorm.weight"]
    param_group = [
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (not any(nd in n for nd in no_decay))],
         'weight_decay': arg.weight_decay},
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (any(nd in n for nd in no_decay))],
         'weight_decay': 0.0},
    ]

    if not arg.prefix_tuning:
        param_group += [
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (not any(nd in n for nd in no_decay))],  # name中不包含bias和LayerNorm.weight
             'weight_decay': arg.weight_decay},
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0},
        ]

    optimizer = AdamW(param_group)  # transformer AdamW
    scheduler = None
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=arg.epoch * data_loader.step_per_epc)

    hyperparams = {
        'batch_size': arg.batch_size,
        'epoch': arg.epoch,
        'identifier': identifier,
        'load_path': arg.load_path,
        'evaluate_every': 1,
        'update_every': 1,
        'load_epoch': arg.load_epoch,
        'load_metric': arg.load_metric,
        'prefix_tuning': arg.prefix_tuning,
        'plm': arg.plm,
        'description': arg.description,
        'neg_rate': neg_rate,
        'add_tokens': arg.add_tokens,
        'p_tuning': arg.p_tuning,
        'rdrop': arg.rdrop,
        'use_structure': arg.use_structure,
        'self_adversarial': arg.self_adversarial,
        'no_use_lm': arg.no_use_lm,
        'contrastive': arg.contrastive,
        'task': arg.task,
        'wandb': arg.wandb
    }

    trainer = Trainer(data_loader, model, lm_tokenizer, optimizer, device, hyperparams)

    if arg.link_prediction:
        trainer.link_prediction(split='test')
    elif arg.triple_classification:
        trainer.triple_classification(split='test')
    else:
        trainer.run()