#!/usr/bin/env python

from __future__ import print_function
import argparse
import os.path as path

import h5py
import numpy as np
import json
import os

PROJ_ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
DATA_ROOT = path.join(PROJ_ROOT, 'data')

def load_layer(name, dataset):
    with open(os.path.join(dataset,name + '.json')) as f_layer:
        return json.load(f_layer)

# =====================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='/data/vision/torralba/health-habits/im2recipe/recipe1M')
parser.add_argument('--vocab', default='vocab.txt')
parser.add_argument('--toks', default='tokenized_instructions_.txt')
parser.add_argument('--max-seqlen', default=40, type=int)
parser.add_argument('--out-suffix', default='')
args = parser.parse_args()
# =====================================================================================

# load vocab
with open(os.path.join(DATA_ROOT,args.vocab),'r') as f_vocab:
    vocab = {w.rstrip(): i+3 for i, w in enumerate(f_vocab)}
    # +1 for lua, UNK, and </r>
    vocab['UNK'] = 1

print('Loading dataset.')
dataset = load_layer('layer1',args.dataset)

print('Assembling tokens.')
max_seqlen = args.max_seqlen - 1 # </s>
recipe_ids = { 'train': [], 'val': [], 'test': [] }
toks_lists = { 'train': [], 'val': [], 'test': [] }
recipe_lens = { 'train': [], 'val': [], 'test': [] }
sent_lens = { 'train': [], 'val': [], 'test': [] }
rbp = { 'train': [], 'val': [], 'test': [] } # recipe base pointers (boundaries)
with open(os.path.join(DATA_ROOT,args.toks),'r') as f_toks:
    for i, tok_recipe in enumerate(f_toks):
        tok_recipe = tok_recipe.strip()

        recipe = dataset[i]
        partition = recipe['partition']

        recipe_ids[partition].append(recipe['id'].encode('utf8'))

        part_rbp = rbp[partition]
        if len(part_rbp) == 0:
            part_rbp.append(1) # 1 for lua
        else:
            rbp[partition].append(rbp[partition][-1] + recipe_lens[partition][-1])

        tok_sents = tok_recipe.split('\t')
        rlen = len(tok_sents)

        recipe_lens[partition].append(rlen)

        part_toks = toks_lists[partition]
        for tok_sent in tok_sents:
            toks = tok_sent.split(' ')[:max_seqlen]
            sent_lens[partition].append(len(toks))
            part_toks.append([vocab.get(t, vocab['UNK']) for t in toks])

recipe_toks = {}
for part, toks_lists in toks_lists.iteritems():
    toks_vec = np.zeros((len(toks_lists), max_seqlen), dtype='uint16')
    for i, toks_list in enumerate(toks_lists):
        toks_vec[i, :len(toks_list)] = toks_list
    recipe_toks[part] = toks_vec

print('Writing out data.')
suff = '_' + args.out_suffix if args.out_suffix else ''
with h5py.File(path.join(DATA_ROOT, 'dataset%s.h5' % suff), 'w') as f_ds:
    for part in recipe_ids:
        f_ds.create_dataset('/ids_%s' % part, data=recipe_ids[part])
        f_ds.create_dataset('/toks_%s' % part, data=recipe_toks[part])
        f_ds.create_dataset('/rlens_%s' % part, data=recipe_lens[part])
        f_ds.create_dataset('/slens_%s' % part, data=sent_lens[part])
        f_ds.create_dataset('/rbps_%s' % part, data=rbp[part])
