import argparse
import pickle
import sys
import os
import random
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn

from utils import count_parameters, determine_op_match, determine_x_match, determine_con_match
from data_loader import DataLoader
from ir_filters_diffs import get_filters_diffs


# parser
parser = argparse.ArgumentParser()
# setup
parser.add_argument('--seed', type=int, default=0xBADB1A5,
                    help='random seed')
parser.add_argument('--data_path', default="data/",
                    help='path to base data directory')
parser.add_argument('--custom_name', default='all',
                    help='name of custom dataset')
parser.add_argument('--vocab_path', default='vocab.pt',
                    help='path to vocab')
parser.add_argument('-bs','--batch_size', type=int, default=256,
                    help='batch size for evaluation')
parser.add_argument('--out_file', default='results/results')
# Analysis
parser.add_argument('--no_all_pairs', action='store_true',
                    help='Dont analyze all_pairs filter (big slowdown)')
parser.add_argument('--all_source_positions', action='store_true',
                    help='Include filters for every source posiiton (e.g. sos, eos, parentheses, etc.)')
parser.add_argument('--return_Q', action='store_true',
                    help="Analyze query vectors")
parser.add_argument('--return_K', action='store_true',
                    help="Analyze key vectors")
parser.add_argument('--return_V', action='store_true',
                    help="Analyze value vectors")
parser.add_argument('--return_R', action='store_true',
                    help="Analyze role vectors")
parser.add_argument('--return_vbar', action='store_true',
                    help="Analyze filler vectors")
parser.add_argument('--return_newv', action='store_true',
                    help="Analyze binding vectors")
parser.add_argument('--return_ff', action='store_true',
                    help="Analyze output of feedforward")
parser.add_argument('--return_mha_res', action='store_true',
                    help="Analyze output of mha + residual connection")
parser.add_argument('--return_ff_res', action='store_true',
                    help="Analyze output of feedforward + residual connection")
parser.add_argument('--include_decoder', action='store_true',
                    help="Include decoder self-attention and decoder-encoder attention")
parser.add_argument('--aggregate_module', default='encoder',
                    help='Module to use for aggregate analyses')
parser.add_argument('--aggregate_rep_type', default='queries',
                    help='Representation type to use for aggregate analyses')
parser.add_argument('--aggregate_layer', type=int, default=5,
                    help='Layer to use for aggregate analyses')
# model
parser.add_argument('--model_name', type=str, choices=['TP','TF'], default='TP',
                    help='model name')
parser.add_argument('--dropout', type=float,
                    default=0.0, 
                    help='dropout (default: 0.0)')
parser.add_argument('--hidden', type=int,
                    default=512, 
                    help='hidden size (default: 512)')
parser.add_argument('-l','--n_layers', type=int,
                    default=6, 
                    help='number of transformer layers (default: 6)')
parser.add_argument('-nh','--n_heads', type=int,
                    default=8,
                    help='number of attention heads (default: 8)')
parser.add_argument('-f','--filter', type=int,
                    default=2048, 
                    help='filter size (default: 2048)')
parser.add_argument('--no_residual', action='store_true',
                    help="No residual connections")
parser.add_argument('--load_model', type=str, default="weights/TP_Transformer.pt",
                    help='Model to load (default: "")')

p = parser.parse_args()

# CUDA
if torch.cuda.is_available():
  p.device = torch.device('cuda')
  n_gpus = torch.cuda.device_count()
else:
  print("WARNING: No CUDA device available. Using CPU.")
  p.device = torch.device('cpu')
  n_gpus = 0

# Random seed
random.seed(p.seed)
np.random.seed(p.seed)
torch.manual_seed(p.seed)
torch.backends.cudnn.deterministic = True

# Load vocabulary
vocab = torch.load(p.vocab_path)
p.input_dim = len(vocab)
p.output_dim = p.input_dim
p.SOS = vocab.stoi['<sos>']  # start of sentence token
p.EOS = vocab.stoi['<eos>']  # end of sentence token
p.PAD = vocab.stoi['<pad>']  # padding token

# Build model by importing the python module
p.d_r = p.hidden // p.n_heads # dimension of role vectors
print("Building model ...")
if p.model_name == 'TP':
  from models.TP_Transformer import build_transformer
  model = build_transformer(params=p, pad_idx=p.PAD).to(p.device)
elif p.model_name == 'TF':
  from models.Transformer import build_transformer
  model = build_transformer(params=p, pad_idx=p.PAD).to(p.device)
else:
  print("{} is not a valid model name.".format(p.model_name))
  sys.exit(1)
print("Done. {} trainable parameters.".format(count_parameters(model)))

# Load model
if p.load_model:
  print("Loading model from {}".format(p.load_model))
  # DataParallel over multiple GPUs
  # need to do this to match keys in state_dict, even if n_gpus=1
  model = nn.DataParallel(model) 
  if p.device == torch.device('cpu'):
    state = torch.load(p.load_model, map_location=torch.device('cpu'))
  else:
    state = torch.load(p.load_model)
  model_state = state['model']
  model.load_state_dict(model_state)
model = model.to(p.device)
print("Done.")

# For doing aggregate analyses across all custom test sets
agg_op_dists = [] # list of all distances between operator vectors
agg_op_diffs = [] # list of all differences between corresponding results
agg_op_match = [] # list of booleans indicating whether operators are matched to results
agg_x_dists = [] # list of all distances between digit vectors
agg_x_diffs = [] # list of all differences between digits
agg_x_match = [] # list of booleans indicating whether digit vectors are matched to digit values
agg_con_dists = [] # list of all distances between vectors in constituents
agg_con_diffs = [] # list of all differences between corresponding results
agg_con_match = [] # list of booleans indicating whether constituents are matched to results


if p.custom_name == 'all':
  custom_names = ['custom_adddiv', 'custom_muldiv', 'custom_addmul', 
                  'custom_addmuldiv', 'custom_adddivadd', 'custom_adddiv2']
else:
  custom_names = [p.custom_name]

for custom_name in custom_names:
  print("Starting custom dataset: {}".format(custom_name))
  # Dataloader for custom examples
  print("using full loader for custom examples")
  custom_dir = 'custom'

  module = DataLoader(module_name=custom_dir,
                      train_bs=p.batch_size,
                      eval_bs=p.batch_size,
                      device=p.device,
                      vocab=vocab,
                      base_folder=p.data_path, 
                      train_name=custom_name, 
                      eval_name=custom_name, 
                      x_ext='.x', 
                      y_ext='.y')
  iterator = module.eval_iterator

  # Lists for collecting statistics/examples
  correct = []
  src_text_samples = [] # List of src_texts [n_samples, src_seq_len]
  trg_text_samples = [] # List of src_texts [n_samples, trg_seq_len]
  src_seq_lens = [] # for reorganizing flattened encoder matrices into samples [n_samples] 
  trg_seq_lens = [] # for reorganizing flattened decoder matrices into samples [n_samples] 

  # Dictionary of representations, organized by representaton type
  representations = {}

  # Encoder
  if p.return_Q:
    representations[('encoder', 'queries')] = [] # [n_samples, n_layers, n_heads, src_seq_len, d_q]
  if p.return_K:
    representations[('encoder', 'keys')] = [] # [n_samples, n_layers, n_heads, src_seq_len, d_k]
  if p.return_V:
    representations[('encoder', 'values')] = [] # [n_samples, n_layers, n_heads, src_seq_len, d_v]
  if p.return_R:
    representations[('encoder', 'role_vecs')] = [] # [n_samples, n_layers, n_heads, src_seq_len, d_r]
  if p.return_vbar:
    representations[('encoder', 'fill_vecs')] = [] # [n_samples, n_layers, n_heads, src_seq_len, d_v]
  if p.return_newv:
    representations[('encoder', 'bind_vecs')] = [] # [n_samples, n_layers, n_heads, src_seq_len, d_v]
  if p.return_ff:
    representations[('encoder', 'feedforward')] = [] # [n_samples, n_layers, src_seq_len, d_x]
  if p.return_mha_res:
    representations[('encoder', 'mha_res')] = [] # [n_samples, n_layers, src_seq_len, d_x]
  if p.return_ff_res:
    representations[('encoder', 'ff_res')] = [] # [n_samples, n_layers, src_seq_len, d_x]

  # Decoder
  if p.include_decoder:
    if p.return_Q:
      representations[('decoder', 'self_attention', 'queries')] = [] 
      representations[('decoder', 'encoder_attention', 'queries')] = []
      # [n_samples, n_layers, n_heads, trg_seq_len, d_q]
    if p.return_K:
      representations[('decoder', 'self_attention', 'keys')] = []
      representations[('decoder', 'encoder_attention', 'keys')] = []
      # [n_samples, n_layers, n_heads, trg_seq_len, d_k]
    if p.return_V:
      representations[('decoder', 'self_attention', 'values')] = []
      representations[('decoder', 'encoder_attention', 'values')] = []
      # [n_samples, n_layers, n_heads, trg_seq_len, d_v]
    if p.return_R:
      representations[('decoder', 'self_attention', 'role_vecs')] = []
      representations[('decoder', 'encoder_attention', 'role_vecs')] = []
      # [n_samples, n_layers, n_heads, trg_seq_len, d_r]
    if p.return_vbar:
      representations[('decoder', 'self_attention', 'fill_vecs')] = []
      representations[('decoder', 'encoder_attention', 'fill_vecs')] = []
      # [n_samples, n_layers, n_heads, trg_seq_len, d_v]
    if p.return_newv:
      representations[('decoder', 'self_attention', 'bind_vecs')] = []
      representations[('decoder', 'encoder_attention', 'bind_vecs')] = []
      # [n_samples, n_layers, n_heads, trg_seq_len, d_v]
    if p.return_ff:
      # (only one feedforward per decoder layer)
      representations[('decoder', 'encoder_attention', 'feedforward')] = []
      # [n_samples, n_layers, trg_seq_len, d_x]

  indexed_by_src = []
  indexed_by_trg = []
  for rep_type in representations.keys():
    if rep_type[0] == 'encoder':
      indexed_by_src.append(rep_type)
    elif rep_type[0] == 'decoder':
      if rep_type[1] == 'self_attention':
        indexed_by_trg.append(rep_type)
      elif rep_type[1] == 'encoder_attention':
        if rep_type[2] in ['keys', 'values']:
          indexed_by_src.append(rep_type)
        else:
          indexed_by_trg.append(rep_type)

  # For dealing with dummy head dimension of headless representations
  rep_types_no_heads = ['feedforward', 'mha_res', 'ff_res']

  # Loop through custom  dataset, collecting representations
  print("Looping through intermediate results data, collecting representations...")
  max_src_len = 0
  model.eval()
  with torch.no_grad():
    samples_collected = 0
    for idx, batch in enumerate(iterator):
      src, trg = batch.src, batch.trg 

      # Useful dims
      batch_size = src.shape[0]
      src_seq_len = src.shape[1]
      if src_seq_len > max_src_len:
        max_src_len = src_seq_len
      src_seq_lens += [src_seq_len for i in range(batch_size)]
      trg_seq_len = trg[:,1:].shape[1] # drop SOS
      trg_seq_lens += [trg_seq_len for i in range(batch_size)]

      # Record src text, trg text
      src_text = [[vocab.itos[i] for i in s] for s in src]
      src_text_samples += src_text
      trg_text = [[vocab.itos[i] for i in s] for s in trg[:,1:]] # drop SOS
      trg_text_samples += trg_text

      # Forward
      src = src.to(p.device)  # [batch_size, src_seq_len]
      trg = trg.to(p.device)  # [batch_size, trg_seq_len]
      logits, reps = model(src, trg[:, :-1])

      # Record predictions, accuracy
      trg_shifted = trg[:,1:] # drop SOS token
      y_hat = torch.argmax(logits, dim=-1)
      prd_text = [[vocab.itos[i] for i in s] for s in y_hat.cpu()]
      matches = (torch.eq(trg_shifted,y_hat) | (trg_shifted==p.PAD)).all(1)
      matches = matches.cpu().numpy().tolist()
      correct += matches

      # Record representations
      for k in representations.keys():
        if k[0] == 'encoder':
          vecs = [reps[k[0]][l][k[1]] for l in range(p.n_layers)]
          # vecs = [n_layers, batch_size, n_heads, src_seq_len, d_*]
          # ff = [n_layers, batch_size, src_seq_len, d_x] (no head dimension for feedforward)
        elif k[0] == 'decoder':
          vecs = [reps['decoder'][l][k[1]][k[2]] for l in range(p.n_layers)]
          # vecs = [n_layers, batch_size, n_heads, src_seq_len, d_*]
          # ff = [n_layers, batch_size, src_seq_len, d_x] (no head dimension for feedforward)

        if k[-1] in rep_types_no_heads:
          vecs = [vs.unsqueeze(1) for vs in vecs] # add dummy head dimension

        # Concatenate by layer
        vecs = [vs.unsqueeze(1) for vs in vecs] # add layer dimension
        vecs = torch.cat(vecs,dim=1)
        # vecs = [batch_size, n_layers, n_heads, src_seq_len, d_*]

        # Undo batching
        vecs = [vs for vs in vecs]
        representations[k] += vecs 
        # representations[k] = n_samples list of [n_layers, n_heads, src_seq_len, d_*]
        # representations['feedforward'] = n_samples list of [n_layers, 1, src_seq_len, d_*]
        
      samples_collected += batch_size 

  print("Done. Collected {} samples".format(samples_collected))
  print("Accuracy: {}".format(np.mean(correct)))

  # Build list of (sample_i, token_i) tuples for identifying token/sample in src sequences
  src_idxs = [] # [n_tokens]
  for sample_i, src_seq_len in enumerate(src_seq_lens):
    for token_i in range(src_seq_len):
      src_idxs.append((sample_i, token_i))
  trg_idxs = [] # [n_tokens]
  for sample_i, trg_seq_len in enumerate(trg_seq_lens):
    for token_i in range(trg_seq_len):
      trg_idxs.append((sample_i, token_i))

  # Functions for filtering pairs of vectors and computing differences
  src_filters, trg_filters, diff_funcs = get_filters_diffs(custom_name, p.no_all_pairs, 
                                                          p.all_source_positions, max_src_len)

  # Compute all distance matrices
  print("Computing all distance matrices")
  distances = {}
  for rep_type, reps in representations.items():
    layers = torch.cat(reps,dim=2) # [n_layers, n_heads, n_samples*src_seq_len, d_*]
    layer_dists = []
    for layer in layers:
      head_dists = []
      for head in layer:
        dists = torch.cdist(head,head) # [n_samples*seq_len, n_samples*seq_len]
        head_dists.append(dists.unsqueeze(0).unsqueeze(1))
      head_dists = torch.cat(head_dists,dim=1)
      layer_dists.append(head_dists)
    layer_dists = torch.cat(layer_dists,dim=0)
    distances[rep_type] = layer_dists.cpu().numpy()
    # distances[rep_type] = [n_layers, n_heads, n_samples*seq_len, n_samples*seq_len]
  print("Done.")

  # Compute differences
  print("Computing differences for each pair of samples")
  ir_diffs = {}
  for diff_name, diff_func in diff_funcs.items():
    diff_mat = np.zeros([samples_collected,samples_collected])
    for si1, src1 in enumerate(src_text_samples):
      for si2, src2 in enumerate(src_text_samples):
        diff = diff_func(src1,src2)
        diff_mat[si1,si2] = diff 
    ir_diffs[diff_name] = diff_mat
    # ir_diffs[diff_name] = [n_samples, n_samples]
  print("Done.")

  # Compute src slices given by each filter
  print("Computing slices for each src filter")
  src_slices = {filt_name:[[],[],[],[]] for filt_name in src_filters.keys()}
  for (idx1, (si1, ti1)), (idx2, (si2, ti2)) in combinations(enumerate(src_idxs), 2):
    src1 = src_text_samples[si1]
    src2 = src_text_samples[si2]
    for filt_name, filt in src_filters.items():
        if filt(ti1, ti2, src1, src2):
          src_slices[filt_name][0].append(idx1)
          src_slices[filt_name][1].append(idx2)
          src_slices[filt_name][2].append(si1)
          src_slices[filt_name][3].append(si2)
  print("Done.")

  # Compute trg slices given by each filter
  print("Computing slices for each trg filter")
  trg_slices = {filt_name:[[],[],[],[]] for filt_name in trg_filters.keys()}
  for (idx1, (si1, ti1)), (idx2, (si2, ti2)) in combinations(enumerate(trg_idxs), 2):
    trg1 = trg_text_samples[si1]
    trg2 = trg_text_samples[si2]
    for filt_name, filt in trg_filters.items():
        if filt(ti1, ti2, trg1, trg2):
          trg_slices[filt_name][0].append(idx1)
          trg_slices[filt_name][1].append(idx2)
          trg_slices[filt_name][2].append(si1)
          trg_slices[filt_name][3].append(si2)
  print("Done.")
        
  # Loop through every pair of points in each rep_type, layer, head, computing correlations
  print("Computing all correlations")
  results = {}
  for rep_type, rep_dists in distances.items():
    print("  Starting rep_type: {}".format(rep_type))
    if rep_type in indexed_by_src:
      slices = src_slices 
    elif rep_type in indexed_by_trg:
      slices = trg_slices
    layer_r = [] # one for each layer
    for l_i,layer_dists in enumerate(rep_dists):
      print("    Starting layer: {}".format(l_i))
      head_r = [] # one for each head
      for h_i, head_dists in enumerate(layer_dists):
        #print("          Starting head: {}".format(h_i))
        filt_r = {} # one for each filter
        for filt_name, filt_slice in slices.items():
          idx1_slice = filt_slice[0]
          idx2_slice = filt_slice[1]
          si1_slice = filt_slice[2]
          si2_slice = filt_slice[3]
          dists = head_dists[idx1_slice, idx2_slice]
          diff_r = {} # one for each diff type
          for diff_name, diff_mat in ir_diffs.items():
            diffs = diff_mat[si1_slice, si2_slice]
            if np.mean(dists) < 1e-3 or np.mean(diffs) < 1e-3: # prevent numerical errors
              # sometimes all distances will be 0 (e.g. 'plus', layer0, Q/K/V/R)
              # sometimes all differences will be 0 (e.g. 'diff_x1', 'plus_left')
              r, p_val = 0.0, 1.0 
            else:
              #r,p_val = pearsonr(dists,diffs)
              r,p_val = spearmanr(dists,diffs)
            # Record for aggregate analyses
            if rep_type == (p.aggregate_module, p.aggregate_rep_type) and l_i == p.aggregate_layer:
              include_op, matched_op = determine_op_match(custom_name, diff_name, filt_name)
              include_x, matched_x = determine_x_match(custom_name, diff_name, filt_name)
              include_con, matched_con = determine_con_match(custom_name, diff_name, filt_name)
              if include_op:
                agg_op_dists.append(dists)
                agg_op_diffs.append(diffs)
                agg_op_match.append(np.array([matched_op for i in range(len(dists))]))
              if include_x:
                agg_x_dists.append(dists)
                agg_x_diffs.append(diffs)
                agg_x_match.append(np.array([matched_x for i in range(len(dists))]))
              if include_con:
                agg_con_dists.append(dists)
                agg_con_diffs.append(diffs)
                agg_con_match.append(np.array([matched_con for i in range(len(dists))]))
            diff_r[diff_name] = (r,p_val)
          filt_r[filt_name] = diff_r 
        head_r.append(filt_r)
      layer_r.append(head_r)
    results[rep_type] = layer_r

  # Save statistics to file using pickle
  out_path = p.out_file + '_' + p.model_name + '_' + custom_name + '.P'
  print("Saving statistics to {}".format(out_path))
  with open(out_path,'wb') as f:
    pickle.dump(results,f)

# Do aggregate analyses: Operations
agg_op_dists = np.expand_dims(np.concatenate(agg_op_dists, axis=0), axis=1)
agg_op_diffs = np.expand_dims(np.concatenate(agg_op_diffs, axis=0), axis=1)
agg_op_match = np.expand_dims(np.concatenate(agg_op_match, axis=0), axis=1)
agg_op_mat = np.concatenate([agg_op_dists, agg_op_match, agg_op_diffs], axis=1)
# Correaltion between dists and diffs: matched operations
agg_op_dists_matched = agg_op_dists[agg_op_match]
agg_op_diffs_matched = agg_op_diffs[agg_op_match]
matched_op_r, matched_op_p = spearmanr(agg_op_dists_matched, agg_op_diffs_matched)
print("Matched operations: ", "Spearman r:, ", matched_op_r, "p = ", matched_op_p)
# Correlation between dists and diffs: unmatched operations
agg_op_dists_unmatched = agg_op_dists[np.logical_not(agg_op_match)]
agg_op_diffs_unmatched = agg_op_diffs[np.logical_not(agg_op_match)]
unmatched_op_r, unmatched_op_p = spearmanr(agg_op_dists_unmatched, agg_op_diffs_unmatched)
print("Unmatched operations: ", "Spearman r:, ", unmatched_op_r, "p = ", unmatched_op_p)

# Do aggregate analyses: digits (x)
agg_x_dists = np.expand_dims(np.concatenate(agg_x_dists, axis=0), axis=1)
agg_x_diffs = np.expand_dims(np.concatenate(agg_x_diffs, axis=0), axis=1)
agg_x_match = np.expand_dims(np.concatenate(agg_x_match, axis=0), axis=1)
agg_x_mat = np.concatenate([agg_x_dists, agg_x_match, agg_x_diffs], axis=1)
# Correaltion between dists and diffs: matched digits
agg_x_dists_matched = agg_x_dists[agg_x_match]
agg_x_diffs_matched = agg_x_diffs[agg_x_match]
matched_x_r, matched_x_p = spearmanr(agg_x_dists_matched, agg_x_diffs_matched)
print("Matched digits: ", "Spearman r:, ", matched_x_r, "p = ", matched_x_p)
# Correlation between dists and diffs: unmatched digits
agg_x_dists_unmatched = agg_x_dists[np.logical_not(agg_x_match)]
agg_x_diffs_unmatched = agg_x_diffs[np.logical_not(agg_x_match)]
unmatched_x_r, unmatched_x_p = spearmanr(agg_x_dists_unmatched, agg_x_diffs_unmatched)
print("Unmatched digits: ", "Spearman r:, ", unmatched_x_r, "p = ", unmatched_x_p)

# Do aggregate analyses: constituents
agg_con_dists = np.expand_dims(np.concatenate(agg_con_dists, axis=0), axis=1)
agg_con_diffs = np.expand_dims(np.concatenate(agg_con_diffs, axis=0), axis=1)
agg_con_match = np.expand_dims(np.concatenate(agg_con_match, axis=0), axis=1)
agg_con_mat = np.concatenate([agg_con_dists, agg_con_match, agg_con_diffs], axis=1)
# Correaltion between dists and diffs: matched digits
agg_con_dists_matched = agg_con_dists[agg_con_match]
agg_con_diffs_matched = agg_con_diffs[agg_con_match]
matched_con_r, matched_con_p = spearmanr(agg_con_dists_matched, agg_con_diffs_matched)
print("Matched constituents: ", "Spearman r:, ", matched_con_r, "p = ", matched_con_p)
# Correlation between dists and diffs: unmatched digits
agg_con_dists_unmatched = agg_con_dists[np.logical_not(agg_con_match)]
agg_con_diffs_unmatched = agg_con_diffs[np.logical_not(agg_con_match)]
unmatched_con_r, unmatched_con_p = spearmanr(agg_con_dists_unmatched, agg_con_diffs_unmatched)
print("Unmatched constituents: ", "Spearman r:, ", unmatched_con_r, "p = ", unmatched_con_p)

# Save matrices for linear regression
agg_op_fn = p.out_file + '_' + p.model_name + '_aggregate_op.csv' 
np.savetxt(agg_op_fn, agg_op_mat, delimiter=",")
agg_x_fn = p.out_file + '_' + p.model_name + '_aggregate_x.csv' 
np.savetxt(agg_x_fn, agg_x_mat, delimiter=",")
agg_con_fn = p.out_file + '_' + p.model_name + '_aggregate_con.csv' 
np.savetxt(agg_con_fn, agg_con_mat, delimiter=",")
