#!/usr/bin/env python

from __future__ import print_function
import argparse
import os.path as path

import h5py
import numpy as np

from utils import Layer
import utils

PROJ_ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
DATA_ROOT = path.join(PROJ_ROOT, 'data')

# =====================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('dataset', 'r1m'), default='r1m')
parser.add_argument('--vocab', default='models/w2v/instructions/instructions_w2v_vocab.txt')
parser.add_argument('--toks', default='tokenized/tokenized_instructions.txt')
parser.add_argument('--max-seqlen', default=40, type=int)

parser.add_argument('--out-suffix', default='')
args = parser.parse_args()
# =====================================================================================

# load vocab
with open(utils.dspath(args.vocab, ds=args.dataset)) as f_vocab:
    vocab = {w.rstrip(): i+4 for i, w in enumerate(f_vocab)}
    # +1 for lua, +1 for UNK, +1 for <r>, +1 for </r>
    vocab['UNK'] = 1

print('Loading dataset.')
dataset = Layer.load(Layer.L1, ds=args.dataset)

print('Assembling tokens.')
recipe_ids = { 'train': [], 'val': [], 'test': [] }
toks_lists = { 'train': [], 'val': [], 'test': [] }
recipe_lens = { 'train': [], 'val': [], 'test': [] }
sent_lens = { 'train': [], 'val': [], 'test': [] }
rbp = { 'train': [], 'val': [], 'test': [] } # recipe base pointers (boundaries)
with open(utils.dspath(args.toks, ds=args.dataset)) as f_toks:
    for i, tok_recipe in enumerate(f_toks):
        tok_recipe = tok_recipe.strip()

        recipe = dataset[i]
        partition = recipe['partition']
        recipe_ids[partition].append(recipe['id'])

        part_rbp = rbp[partition]
        if len(part_rbp) == 0:
            part_rbp.append(1) # 1 for lua
        else:
            rbp[partition].append(rbp[partition][-1] + recipe_lens[partition][-1])

        tok_sents = tok_recipe.split('\t')
        rlen = len(recipe['instructions'])
        assert len(tok_sents) == rlen

        recipe_lens[partition].append(rlen)

        part_toks = toks_lists[partition]
        for tok_sent in tok_sents:
            toks = tok_sent.split(' ')[:args.max_seqlen-2] # </s> toks... </s>
            sent_lens[partition].append(len(toks))
            part_toks.append([vocab.get(t, vocab['UNK']) for t in toks])

recipe_toks = {}
for part, toks_lists in toks_lists.iteritems():
    toks_vec = np.zeros((len(toks_lists), args.max_seqlen-2), dtype='uint16')
    for i, toks_list in enumerate(toks_lists):
        toks_vec[i, :len(toks_list)] = toks_list
    recipe_toks[part] = toks_vec

print('Writing out data.')
if args.out_suffix: suff = '_' + args.out_suffix
with h5py.File(path.join(DATA_ROOT, 'dataset%s.h5' % args.out_suffix), 'w') as f_ds:
    for part in recipe_ids:
        f_ds.create_dataset('/ids_%s' % part, data=recipe_ids[part])
        f_ds.create_dataset('/toks_%s' % part, data=recipe_toks[part])
        f_ds.create_dataset('/rlens_%s' % part, data=recipe_lens[part])
        f_ds.create_dataset('/slens_%s' % part, data=sent_lens[part])
        f_ds.create_dataset('/rbps_%s' % part, data=rbp[part])
