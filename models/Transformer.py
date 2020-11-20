import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_vis_params(params):
  vis_params = {}
  if hasattr(params,'return_attention'):
    vis_params['return_attention'] = params.return_attention
  else:
    vis_params['return_attention'] = False
  if hasattr(params,'return_norms'):
    vis_params['return_norms'] = params.return_norms
  else:
    vis_params['return_norms'] = False
  if hasattr(params,'return_Q'):
    vis_params['return_Q'] = params.return_Q
  else:
    vis_params['return_Q'] = False
  if hasattr(params,'return_K'):
    vis_params['return_K'] = params.return_K
  else:
    vis_params['return_K'] = False
  if hasattr(params,'return_V'):
    vis_params['return_V'] = params.return_V
  else:
    vis_params['return_V'] = False
  if hasattr(params,'return_vbar'):
    vis_params['return_vbar'] = params.return_vbar
  else:
    vis_params['return_vbar'] = False
  if hasattr(params,'return_ff'):
    vis_params['return_ff'] = params.return_ff
  else:
    vis_params['return_ff'] = False
  if hasattr(params,'return_mha_res'):
    vis_params['return_mha_res'] = params.return_mha_res
  else:
    vis_params['return_mha_res'] = False
  if hasattr(params,'return_ff_res'):
    vis_params['return_ff_res'] = params.return_ff_res
  else:
    vis_params['return_ff_res'] = False
  if hasattr(params,'return_embeddings'):
    vis_params['return_embeddings'] = params.return_embeddings
  else:
    vis_params['return_embeddings'] = False
  return vis_params

def build_encoder_layer(params):
  # for probe classifiers in tpx/probe/probing_classifier.py
  vis_params = get_vis_params(params)
  encoderlayer = EncoderLayer(hid_dim = params.hidden,
                              n_heads = params.n_heads,
                              pf_dim = params.filter,
                              self_attention = SelfAttention,
                              positionwise_feedforward = PositionwiseFeedforward,
                              dropout = params.dropout,
                              vis_params = vis_params)
  return encoderlayer

def build_transformer(params, pad_idx):

  print("starting build_transformer...", flush=True)

  vis_params = get_vis_params(params)

  embedding = TokenEmbedding(d_vocab=params.input_dim,
                             d_h=params.hidden,
                             d_p=params.hidden,
                             dropout=params.dropout,
                             max_length=200)

  encoder = Encoder(hid_dim=params.hidden,
                    n_layers=params.n_layers,
                    n_heads=params.n_heads,
                    pf_dim=params.filter,
                    encoder_layer=EncoderLayer,
                    self_attention=SelfAttention,
                    positionwise_feedforward=PositionwiseFeedforward,
                    dropout=params.dropout,
                    vis_params=vis_params)

  decoder = Decoder(hid_dim=params.hidden,
                    n_layers=params.n_layers,
                    n_heads=params.n_heads,
                    pf_dim=params.filter,
                    decoder_layer=DecoderLayer,
                    self_attention=SelfAttention,
                    positionwise_feedforward=PositionwiseFeedforward,
                    dropout=params.dropout,
                    vis_params=vis_params)

  model =  Seq2Seq(embedding=embedding,
                   encoder=encoder,
                   decoder=decoder,
                   pad_idx=pad_idx,
                   vis_params=vis_params)

  print("returning model...", flush=True)

  return model

class BeamPath():
    def __init__(self, beam_path=[], eos_index=None, max_len=99, context={}):
        if beam_path:
            self.path = list(beam_path.path)    
            self.eos_index = beam_path.eos_index
            self.max_len = beam_path.max_len
            self.context = dict(beam_path.context)
            self.score = self._compute_score()
        else:
            self.path = []
            self.eos_index = eos_index
            self.max_len = max_len
            self.context = context
            self.score = 0

    def append(self, prob, index):
        #print("BeamPath.append")
        pair = {"prob": prob, "index": index}
        self.path.append(pair)
        self._compute_score()

    def _compute_score(self):
        self.score = np.prod( [pair["prob"] for pair in self.path] )
        return self.score

    def done(self):
        return self.path and (len(self.path) >= self.max_len or self.path[-1]["index"] == self.eos_index)

def top_indexes_and_values(array, N):
    indexes = np.flip(np.argsort(array))[:N]
    values = np.array(array)[indexes]
    return indexes, values

def take_best_paths(paths, N):
    #print("take_best_paths")

    scores = [path.score for path in paths]
    indexes, _ = top_indexes_and_values(scores, N)
    best_paths = np.array(paths)[indexes]
    return best_paths

def make_top_subpaths(path, N, probs):
    #print("make_top_subpaths")

    subpaths = []
    indexes, values = top_indexes_and_values(probs, N)
    for index, prob in zip(indexes, values):
        sub = BeamPath(path)
        sub.append(prob, index)
        subpaths.append(sub)

    return subpaths

def beam_search(predict_next_token, context, eos_index, max_len, N=3):
    #paths = [BeamPath(max_len=max_len, eos_index=eos_index) for _ in range(N)]
    paths = [BeamPath(max_len=max_len, eos_index=eos_index, context=context)]
    outer_count = 0
    predict_count = 0

    while True:
        all_subs = []
        expanded = False
        outer_count += 1

        for path in paths:
            if not path.done():
                probs = predict_next_token(path.context, predict_count)
                predict_count += 1
                subpaths = make_top_subpaths(path, N, probs)
                all_subs += subpaths
                expanded = True
            else:
                all_subs += [path]

        if not expanded:
            # no more paths to expand
            break

        paths = take_best_paths(all_subs, N)

    #print("outer_count=", outer_count)
    best_path = take_best_paths(paths, 1)[0]
    return best_path

class TokenEmbedding(nn.Module):
  def __init__(self, d_vocab, d_h, d_p, dropout, max_length):
    super(TokenEmbedding, self).__init__()
    self.dropout = nn.Dropout(dropout)

    # token encodings
    self.d_h = d_h
    self.tok_embedding = nn.Embedding(d_vocab, d_h)
    self.scale = torch.sqrt(torch.FloatTensor([d_h]))

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_length, d_p)
    position = torch.arange(0., max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_p, 2) *
                         -(math.log(10000.0) / d_p))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
    # pe = [1, seq_len, d_p]

    self.reset_parameters()  # init tok_embedding to N(0,1/sqrt(d_h))

  def forward(self, src):
    # src = [batch_size, src_seq_len]

    # scale up embedding to be N(0,1)
    tok_emb = self.tok_embedding(src) * self.scale.to(src.device)
    pos_emb = torch.autograd.Variable(self.pe[:, :src.size(1)],
                                      requires_grad=False)
    x = tok_emb + pos_emb
    x = self.dropout(x)

    # src = [batch_size, src_seq_len, d_h]
    return x

  def transpose_forward(self, trg):
    # trg = [batch_size, trg_seq_len, d_h]
    logits = torch.einsum('btd,vd->btv',trg,self.tok_embedding.weight)
    # logits = torch.matmul(trg, torch.transpose(self.tok_embedding.weight, 0, 1))
    # logits = [batch_size, trg_seq_len, d_vocab]
    return logits

  def reset_parameters(self):
    nn.init.normal_(self.tok_embedding.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_h))


class Encoder(nn.Module):
  def __init__(self, hid_dim, n_layers, n_heads, pf_dim,
               encoder_layer, self_attention, positionwise_feedforward, 
               dropout, vis_params):
    super().__init__()

    self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim,
                                               self_attention,
                                               positionwise_feedforward,
                                               dropout,vis_params)
                                 for _ in range(n_layers)])


  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_len]
    # src_mask = [batch_size, src_seq_len]
    reps = []
    for layer in self.layers:
      src, reps_l = layer(src, src_mask)
      reps.append(reps_l)

    return src, reps


class EncoderLayer(nn.Module):
  def __init__(self, hid_dim, n_heads, pf_dim, self_attention,
               positionwise_feedforward, dropout, vis_params):
    super().__init__()
    self.return_ff = vis_params['return_ff']
    self.return_mha_res = vis_params['return_mha_res']
    self.return_ff_res = vis_params['return_ff_res']

    self.layernorm1 = nn.LayerNorm(hid_dim)
    self.layernorm2 = nn.LayerNorm(hid_dim)
    self.layernorm3 = nn.LayerNorm(hid_dim)
    self.MHA = self_attention(hid_dim, n_heads, dropout, vis_params)
    self.densefilter = positionwise_feedforward(hid_dim, pf_dim, dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)


  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_size, hid_dim]
    # src_mask = [batch_size, src_seq_size]

    # sublayer 1
    z = self.layernorm1(src)
    z, reps = self.MHA(z, z, z, src_mask)
    z = self.dropout1(z)
    mha_res = src + z

    # sublayer 2
    z = self.layernorm2(mha_res)
    ff = self.densefilter(z)
    z = self.dropout2(ff)
    ff_res = mha_res + z

    if self.return_ff:
      reps['feedforward'] = ff.detach()
    if self.return_mha_res:
      reps['mha_res'] = mha_res.detach()
    if self.return_ff_res:
      reps['ff_res'] = ff_res.detach()

    return self.layernorm3(ff_res), reps


class SelfAttention(nn.Module):
  def __init__(self, hid_dim, n_heads, dropout, vis_params):
    super().__init__()

    self.hid_dim = hid_dim
    self.n_heads = n_heads

    assert hid_dim % n_heads == 0

    self.w_q = nn.Linear(hid_dim, hid_dim)
    self.w_k = nn.Linear(hid_dim, hid_dim)
    self.w_v = nn.Linear(hid_dim, hid_dim)

    self.linear = nn.Linear(hid_dim, hid_dim)
    self.dropout = nn.Dropout(dropout)
    self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    self.reset_parameters()

    self.return_attention = vis_params['return_attention']
    self.return_norms = vis_params['return_norms']
    self.return_Q = vis_params['return_Q']
    self.return_K = vis_params['return_K']
    self.return_V = vis_params['return_V']
    self.return_vbar = vis_params['return_vbar']

  def forward(self, query, key, value, mask=None):
    # query = key = value = [batch_size, seq_len, hid_dim]
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    bsz = query.shape[0]
    src_len = key.shape[1]
    trg_len = query.shape[1]

    Q = self.w_q(query)
    K = self.w_k(key)
    V = self.w_v(value)
    # Q, K, V = [batch_size, seq_len, hid_dim]

    Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)\
         .permute(0,2,1,3)
    K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)\
         .permute(0,2,1,3)
    V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)\
         .permute(0,2,1,3)
    # Q, K, V = [batch_size, n_heads, seq_size, hid_dim // n heads]
    d_v = V.shape[3]

    energy = torch.einsum('bhid,bhjd->bhij',Q,K) / self.scale.to(key.device)
    # energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(key.device)

    # energy   = [batch_size, n_heads, query_pos     , key_pos]
    # src_mask = [batch_size, 1      , 1             , attn]
    # trg_mask = [batch_size, 1      , query_specific, attn]

    if mask is not None:
      energy = energy.masked_fill(mask == 0, -1e10)

    attention = self.dropout(F.softmax(energy, dim=-1))
    # attention = [batch_size, n_heads, seq_size, seq_size]

    # Get norms of weighted value vectors (for analysis)
    V_ = V.expand(trg_len,bsz,self.n_heads,src_len,d_v).permute(1,2,0,3,4)
    A_ = attention.expand(d_v,bsz,self.n_heads,trg_len,src_len).permute(1,2,3,4,0)
    norms = torch.norm(A_*V_,dim=4)

    v_bar = torch.einsum('bhjd,bhij->bhid',V,attention)
    # x = torch.matmul(attention, V)
    # x = [batch_size, n_heads, seq_size, hid_dim // n heads]

    x = v_bar.permute(0, 2, 1, 3).contiguous()
    # x = [batch_size, seq_size, n_heads, hid_dim // n heads]

    x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
    # x = [batch_size, src_seq_size, hid_dim]

    x = self.linear(x)
    # x = [batch_size, seq_size, hid_dim]

    reps = {}
    if self.return_attention:
      reps['attention'] = attention.detach()
    if self.return_norms:
      reps['norms'] = norms.detach()
    if self.return_Q:
      reps['queries'] = Q.detach()
    if self.return_K:
      reps['keys'] = K.detach()
    if self.return_V:
      reps['values'] = V.detach()
    if self.return_vbar:
      reps['fill_vecs'] = v_bar.detach()

    return x, reps

  def reset_parameters(self):
    # nn.init.xavier_normal_(self.w_q.weight)
    # nn.init.xavier_normal_(self.w_k.weight)
    # nn.init.xavier_normal_(self.w_v.weight)
    # nn.init.xavier_normal_(self.linear.weight)
    nn.init.xavier_uniform_(self.w_q.weight)
    nn.init.xavier_uniform_(self.w_k.weight)
    nn.init.xavier_uniform_(self.w_v.weight)
    nn.init.xavier_uniform_(self.linear.weight)


class PositionwiseFeedforward(nn.Module):
  def __init__(self, hid_dim, pf_dim, dropout):
    super().__init__()

    self.hid_dim = hid_dim
    self.pf_dim = pf_dim

    self.linear1 = nn.Linear(hid_dim, pf_dim)
    self.linear2 = nn.Linear(pf_dim, hid_dim)
    self.dropout = nn.Dropout(dropout)

    self.reset_parameters()

  def forward(self, x):
    # x = [batch_size, seq_size, hid_dim]

    x = self.linear1(x)
    x = self.dropout(F.relu(x))
    x = self.linear2(x)

    # x = [batch_size, seq_size, hid_dim]
    return x

  def reset_parameters(self):
    #nn.init.kaiming_normal_(self.linear1.weight, a=math.sqrt(5))
    #nn.init.xavier_normal_(self.linear2.weight)
    nn.init.xavier_uniform_(self.linear1.weight)
    nn.init.xavier_uniform_(self.linear2.weight)


class Decoder(nn.Module):
  def __init__(self, hid_dim, n_layers, n_heads, pf_dim, decoder_layer,
               self_attention, positionwise_feedforward, dropout, vis_params):
    super().__init__()

    self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim,
                                               self_attention,
                                               positionwise_feedforward,
                                               dropout, vis_params)
                                 for _ in range(n_layers)])

  def forward(self, trg, src, trg_mask, src_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]
    reps = []
    for layer in self.layers:
      trg, reps_l = layer(trg, src, trg_mask, src_mask)
      reps.append(reps_l)

    return trg, reps


class DecoderLayer(nn.Module):
  def __init__(self, hid_dim, n_heads, pf_dim, self_attention,
               positionwise_feedforward, dropout, vis_params):
    super().__init__()
    self.return_ff = vis_params['return_ff']

    self.layernorm1 = nn.LayerNorm(hid_dim)
    self.layernorm2 = nn.LayerNorm(hid_dim)
    self.layernorm3 = nn.LayerNorm(hid_dim)
    self.layernorm4 = nn.LayerNorm(hid_dim)
    self.selfAttn = self_attention(hid_dim, n_heads, dropout, vis_params)
    self.encAttn = self_attention(hid_dim, n_heads, dropout, vis_params)
    self.densefilter = positionwise_feedforward(hid_dim, pf_dim, dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, trg, src, trg_mask, src_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]

    # self attention
    z = self.layernorm1(trg)
    z, sa_reps = self.selfAttn(z, z, z, trg_mask)
    z = self.dropout1(z)
    trg = trg + z

    # encoder attention
    z = self.layernorm2(trg)
    z, ea_reps = self.encAttn(z, src, src, src_mask)
    z = self.dropout2(z)
    trg = trg + z

    # dense filter
    z = self.layernorm3(trg)
    ff = self.densefilter(z)
    z = self.dropout3(ff)
    trg = trg + z

    if self.return_ff:
      ea_reps['feedforward'] = ff
    reps = {'self_attention':sa_reps,
            'encoder_attention':ea_reps}

    return self.layernorm4(trg), reps


class Seq2Seq(nn.Module):
  def __init__(self, embedding, encoder, decoder, pad_idx, vis_params):
    super().__init__()

    self.embedding = embedding
    self.encoder = encoder
    self.decoder = decoder
    self.pad_idx = pad_idx

    self.return_embeddings = vis_params['return_embeddings']

  def make_masks(self, src, trg):
    # src = [batch_size, src_seq_size]
    # trg = [batch_size, trg_seq_size]

    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
    # trg_mask = [batch_size, 1, trg_seq_size, 1]
    trg_len = trg.shape[1]

    # trg_sub_mask = torch.tril(
    #   torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))
    # trg_mask = trg_pad_mask & trg_sub_mask

    if getattr(torch, "bool") and torch.__version__ != "1.2.0":
      # bug in torch 1.3.0 needs this workaround
      trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool, device=trg.device))
      trg_mask = trg_pad_mask & trg_sub_mask
    else:
      # this is the correct code (torch 1.2.0 and torch 1.4.0?)
      # workarond for torch.tril() not currently supporting bool types
      trg_sub_mask = torch.tril(
        torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))
      trg_mask = trg_pad_mask & trg_sub_mask.bool()

    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]
    return src_mask, trg_mask

  def forward(self, src, trg):
    # src = [batch_size, src_seq_size]
    # trg = [batch_size, trg_seq_size]

    src_mask, trg_mask = self.make_masks(src, trg)
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    src = self.embedding(src)
    trg = self.embedding(trg)
    # src = [batch_size, src_seq_size, hid_dim]

    enc_src, enc_reps = self.encoder(src, src_mask)
    # enc_src = [batch_size, src_seq_size, hid_dim]

    out, dec_reps = self.decoder(trg, enc_src, trg_mask, src_mask)
    # out = [batch_size, trg_seq_size, hid_dim]

    logits = self.embedding.transpose_forward(out)
    # logits = [batch_size, trg_seq_size, d_vocab]

    reps = {'encoder':enc_reps,
            'decoder':dec_reps}

    if self.return_embeddings:
      reps['src_embeddings'] = src
      reps['trg_embeddings'] = trg

    return logits, reps


  def make_src_mask(self, src):
    # src = [batch size, src sent len]
    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

  def make_trg_mask(self, trg):
    # trg = [batch size, trg sent len]
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(
      torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

    #trg_mask = trg_pad_mask & trg_sub_mask
    trg_mask = trg_pad_mask & trg_sub_mask.bool()

    return trg_mask

  def greedy_inference(self, src, sos_idx, eos_idx, max_length, device):
    self.eval()
    src = src.to(device)
    src_mask = self.make_src_mask(src)
    src_emb = self.embedding(src)

    # run encoder
    enc_src, enc_record = self.encoder(src_emb, src_mask)
    trg = torch.ones(src.shape[0], 1).fill_(sos_idx).type_as(src).to(device)

    done = torch.zeros(src.shape[0]).type(torch.uint8).to(device)
    for _ in range(max_length):
      trg_emb = self.embedding(trg)
      trg_mask = self.make_trg_mask(trg)
      # run decoder
      output, dec_record = self.decoder(src=enc_src, trg=trg_emb,
                             src_mask=src_mask, trg_mask=trg_mask)
      logits = self.embedding.transpose_forward(output)
      pred = torch.argmax(logits[:,[-1],:], dim=-1)
      trg = torch.cat([trg, pred], dim=1)

      eos_match = (pred.squeeze(1) == eos_idx)
      #done = done | eos_match
      done = done | eos_match.byte()

      if done.sum() == src.shape[0]:
        break

    return trg, enc_record + dec_record