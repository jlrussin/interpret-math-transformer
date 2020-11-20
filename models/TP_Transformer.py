import os
import math
import pdb
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import update_vis_params, HyperParams

LOG_TERMINAL = False  # big slow down if enabled

def is_nan_string(x):
  if not LOG_TERMINAL:
    return "skip"
  return "NaN" if torch.isnan(x).any() else "ok"

def is_nan(d):
  if LOG_TERMINAL:
    for t in d:
      if torch.isnan(t).any():
        return True
  return False

def debug(*args):
  if LOG_TERMINAL:
    print(*args)
  
def build_encoder_layer(params):
  p = HyperParams()
  p.d_f = params.filter 
  p.n_I = params.n_heads 
  p.d_x = params.hidden 
  p.d_v = p.d_x // p.n_I 
  p.d_r = p.d_x // p.n_I 
  p.d_k = p.d_x // p.n_I 
  p.d_q = p.d_x // p.n_I 
  p.dropout = params.dropout 
  update_vis_params(params,p)
  encoderlayer = EncoderLayer(p)
  return encoderlayer

def build_transformer(params, pad_idx):
  p = HyperParams()
  p.d_vocab = params.input_dim
  p.d_pos = 200 # max input size

  p.d_f = params.filter

  p.n_L = params.n_layers
  p.n_I = params.n_heads

  p.d_x = params.hidden  # token embedding dimension
  p.d_p = params.hidden  # position embedding dimension

  p.d_v = p.d_x // p.n_I  # value dimension
  p.d_r = p.d_x // p.n_I  # role dimension

  p.d_k = p.d_x // p.n_I  # key dimension
  p.d_q = p.d_x // p.n_I  # query dimension

  p.dropout = params.dropout

  update_vis_params(params,p)

  embedding = EmbeddingMultilinearSinusoidal(d_vocab=params.input_dim,
                                             d_x=p.d_x,
                                             d_r=p.d_r,
                                             dropout=params.dropout,
                                             max_length=200)
  encoder = Encoder(p=p)
  decoder = Decoder(p=p)
  model = Seq2Seq(p=p,
                  embedding=embedding,
                  encoder=encoder,
                  decoder=decoder,
                  pad_idx=pad_idx)

  return model


class Encoder(nn.Module):
  def __init__(self, p):
    super().__init__()

    layers = [EncoderLayer(p)]
    for _ in range(p.n_L - 1):
      layers.append(EncoderLayer(p))
    self.layers = nn.ModuleList(layers)

  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_len]
    # src_mask = [batch_size, 1, attn_size]
    debug("\nencoder")
    debug("src: ", src.shape, is_nan_string(src))
    debug("src_mask: ", src_mask.shape, is_nan_string(src_mask))
    pdb.set_trace() if is_nan([src, src_mask]) else ""

    reps = []
    for layer in self.layers:
      src, reps_l = layer(src, src_mask)
      reps.append(reps_l)

    return src, reps


class EncoderLayer(nn.Module):
  def __init__(self, p):
    super().__init__()
    self.return_ff = p.return_ff
    self.return_mha_res = p.return_mha_res 
    self.return_ff_res = p.return_ff_res
    # d_in list of input dimension of z e.g. [p.d_x, p.d_p] or [p.d_v, p.d_r]
    d_h = p.d_x

    # sublayer 1
    self.layernorm1 = nn.LayerNorm(d_h)
    self.MHA = SelfAttention(p)
    self.dropout1 = nn.Dropout(p.dropout)
    # sublayer 2
    self.layernorm2 = nn.LayerNorm(d_h)
    self.densefilter = PositionwiseFeedforward(hid_dim=d_h,
                                               pf_dim=p.d_f,
                                               dropout=p.dropout)
    self.dropout2 = nn.Dropout(p.dropout)
    # output
    self.layernorm3 = nn.LayerNorm(d_h)

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


class EmbeddingMultilinear(nn.Module):
  def __init__(self, d_vocab, d_x, d_p, dropout, max_length, EOS_idx):
    super(EmbeddingMultilinear, self).__init__()
    # token encodings
    self.d_x = d_x
    self.d_p = d_p
    self.max_length = max_length
    self.EOS_idx = EOS_idx
    self.tok_embedding = nn.Embedding(d_vocab, d_x)
    self.pos_embedding = nn.Embedding(max_length, d_p)
    self.pos_linear = nn.Linear(1, d_p)
    self.emb_scale = torch.FloatTensor([math.sqrt(d_x)])
    self.mul_scale = torch.FloatTensor([1./math.sqrt(math.sqrt(2) - 1)])
    self.dropout = nn.Dropout(dropout)
    self.reset_parameters()

  def forward(self, src):
    # src = [batch_size, src_seq_len]

    tok_emb = self.tok_embedding(src) * self.emb_scale.to(src.device)
    # tok_emb = [batch_size, src_seq_len, d_h]

    # id based learned embedding (order info is lost)
    pos_id = torch.arange(0, src.shape[1]).repeat(src.shape[0], 1)
    pos_id_emb = self.pos_embedding(pos_id.to(src.device))

    # linear map of continuous position [0,1]
    pos_cont = pos_id.float() / self.max_length
    pos_cont_emb = self.pos_linear(pos_cont.unsqueeze(-1).to(src.device))
    pos_emb = 0.5*(pos_id_emb + pos_cont_emb)
    # pos_emb = [batch_size, src_seq_len, d_p]

    # such that x below is mean=1 and var=1
    # var(x) = var(a)var(b)+var(a)+var(b)
    tok_emb_scaled = tok_emb * self.mul_scale.to(src.device) + 1
    pos_emb_scaled = pos_emb * self.mul_scale.to(src.device) + 1

    # b batch size
    # l sequence length
    # x token dimension
    # p position dimension
    x = torch.einsum('blx,blp->blxp', tok_emb_scaled, pos_emb_scaled) - 1
    x = self.dropout(x)

    pdb.set_trace() if is_nan([x]) else ""

    # src = [batch_size, src_seq_len, d_x, d_p]
    return x

  def transpose_forward(self, trg):
    # trg = [batch_size, trg_seq_len, d_x]
    logits = torch.matmul(trg, torch.transpose(self.tok_embedding.weight, 0, 1))
    # logits = [batch_size, trg_seq_len, d_vocab]
    return logits

  def reset_parameters(self):
    """
    limit = math.sqrt(3)  # uniform with var=1
    l1 = limit / math.sqrt(self.d_x)  # reduce to var=1/d_x
    l2 = limit / math.sqrt(self.d_p)  # reduce to var=1/d_p

    nn.init.uniform_(self.tok_embedding.weight, a=-l1, b=l1)
    nn.init.uniform_(self.pos_embedding.weight, a=-limit, b=limit)
    nn.init.uniform_(self.pos_linear.weight, a=-l2, b=l2)
    """
    nn.init.normal_(self.tok_embedding.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_x))
    nn.init.normal_(self.pos_embedding.weight,
                    mean=0,
                    std=1)
    nn.init.normal_(self.pos_linear.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_p))


class EmbeddingMultilinearSinusoidal(nn.Module):
  def __init__(self, d_vocab, d_x, d_r, dropout, max_length):
    super(EmbeddingMultilinearSinusoidal, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.max_length = max_length
    self.d_x = d_x

    # token encodings
    self.tok_embedding = nn.Embedding(d_vocab, d_x)
    self.scale = torch.sqrt(torch.FloatTensor([d_x]))

    # pos embedding
    #self.pos_embedding = nn.Embedding(max_length, d_x)

    # sinusoidal encoding
    pe = torch.zeros(max_length, d_x)
    position = torch.arange(0., max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_x, 2) *
                         -(math.log(10000.0) / d_x))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
    # pe = [1, seq_len, d_p]

    # x -> r
    self.linear = nn.Linear(d_x, d_x)
    self.mul_scale = torch.FloatTensor([1. / math.sqrt(math.sqrt(2) - 1)])

    self.reset_parameters()

  def forward(self, src):
    # src = [batch_size, src_seq_len]
    tok_emb = self.tok_embedding(src) * self.scale.to(src.device)

    # id based learned embedding (order info is lost)
    #pos_id = torch.arange(0, src.shape[1]).repeat(src.shape[0], 1)
    #pos_id_emb = self.pos_embedding(pos_id.to(src.device))

    # sinusoidal
    pos_sin_emb = torch.autograd.Variable(self.pe[:, :src.size(1)],
                                          requires_grad=False)
    x = tok_emb + pos_sin_emb # 0.5 * pos_sin_emb + 0.5 * pos_id_emb
    # x = [batch_size, src_seq_len, d_x]

    r = self.linear(x) + 1 # ~N(1,1)
    # r = [batch_size, src_seq_len, d_r]

    # b batch size
    # t sequence length
    # x token dimension
    # r role dimension
    #x = self.mul_scale.to(src.device) * x + 1.0
    #r = self.mul_scale.to(src.device) * r + 1.0
    z = x * r
    # z = torch.einsum('blx,blr->blxr', x, r).view(x.shape[0], x.shape[1], -1)
    #z = z - 1
    z = self.dropout(z)
    # src = [batch_size, src_seq_len, d_x*d_r]
    return z

  def transpose_forward(self, trg):
    # trg = [batch_size, trg_seq_len, d_v, d_r]

    logits = torch.matmul(trg, torch.transpose(self.tok_embedding.weight, 0, 1))
    # logits = [batch_size, trg_seq_len, d_vocab
    return logits

  def reset_parameters(self):
    nn.init.normal_(self.tok_embedding.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_x))
    #nn.init.normal_(self.pos_embedding.weight,
    #                mean=0,
    #                std=1)
    nn.init.normal_(self.linear.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_x))


class SelfAttention(nn.Module):
  def __init__(self, p):
    super().__init__()
    self.p = p
    self.d_h = p.d_x
    self.n_I = p.n_I

    self.W_q = nn.Linear(self.d_h, p.d_q * p.n_I)
    self.W_k = nn.Linear(self.d_h, p.d_k * p.n_I)
    self.W_v = nn.Linear(self.d_h, p.d_v * p.n_I)
    self.W_r = nn.Linear(self.d_h, p.d_r * p.n_I)

    self.W_o = nn.Linear(p.d_v * p.n_I, p.d_x)

    self.dropout = nn.Dropout(p.dropout)
    self.dot_scale = torch.FloatTensor([math.sqrt(p.d_k)])
    self.mul_scale = torch.FloatTensor([1./math.sqrt(math.sqrt(2) - 1)])

    self.return_attention = p.return_attention 
    self.return_norms = p.return_norms 
    self.return_Q = p.return_Q
    self.return_K = p.return_K
    self.return_V = p.return_V
    self.return_R = p.return_R
    self.return_vbar = p.return_vbar
    self.return_newv = p.return_newv 

  def forward(self, query, key, value, mask=None):
    # query = key = value = [batch_size, seq_len, hid_dim]
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    bsz = query.shape[0]
    src_len = key.shape[1]
    trg_len = query.shape[1]

    Q = self.W_q(query)
    K = self.W_k(key)
    V = self.W_v(value)
    R = self.W_r(query)
    # Q, K, V, R = [batch_size, seq_len, n_heads * d_*]

    Q = Q.view(bsz, -1, self.n_I, self.p.d_q).permute(0,2,1,3)
    K = K.view(bsz, -1, self.n_I, self.p.d_k).permute(0,2,1,3)
    V = V.view(bsz, -1, self.n_I, self.p.d_v).permute(0,2,1,3)
    R = R.view(bsz, -1, self.n_I, self.p.d_r).permute(0,2,1,3)
    # Q, K, V, R = [batch_size, n_heads, seq_size, d_*]
    d_v = V.shape[3]

    dot = torch.einsum('bhid,bhjd->bhij', Q, K) / self.dot_scale.to(key.device)
    # energy   = [batch_size, n_heads, query_pos     , key_pos]
    # src_mask = [batch_size, 1      , 1             , attn]
    # trg_mask = [batch_size, 1      , query_specific, attn]

    if mask is not None:
      dot = dot.masked_fill(mask == 0, -1e10)

    attention = self.dropout(F.softmax(dot, dim=-1))
    # attention = [batch_size, n_heads, seq_size, seq_size]

    # Get norms of weighted value vectors (for analysis)
    V_ = V.expand(trg_len,bsz,self.n_I,src_len,d_v).permute(1,2,0,3,4)
    A_ = attention.expand(d_v,bsz,self.n_I,trg_len,src_len).permute(1,2,3,4,0)
    norms = torch.norm(A_*V_,dim=4)

    v_bar = torch.einsum('bhjd,bhij->bhid', V, attention)
    # v_bar = [batch_size, n_heads, seq_size, d_v]

    # bind
    #R = self.mul_scale.to(key.device) * R + 1
    #v_bar = self.mul_scale.to(key.device) * v_bar + 1
    new_v = v_bar * R
    #new_v = new_v - 1
    x = new_v.permute(0,2,1,3).contiguous()
    # v_bar = [batch_size, seq_size, n_heads, d_v]

    x = x.view(bsz, -1, self.n_I * self.p.d_v)
    # new_v = [batch_size, src_seq_size, n_heads * d_v]

    x = self.W_o(x)
    # x = [batch_size, seq_size, d_x]

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
    if self.return_R:
      reps['role_vecs'] = R.detach()
    if self.return_vbar:
      reps['fill_vecs'] = v_bar.detach()
    if self.return_newv:
      reps['bind_vecs'] = new_v.detach()

    return x, reps

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.W_q.weight)
    nn.init.xavier_uniform_(self.W_k.weight)
    nn.init.xavier_uniform_(self.W_v.weight)
    nn.init.xavier_uniform_(self.W_o.weight)

    nn.init.normal_(self.W_r.weight,
                    mean=0,
                    std=1./math.sqrt(self.p.d_r))


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
  def __init__(self, p):
    super().__init__()

    self.layers = nn.ModuleList([DecoderLayer(p) for _ in range(p.n_L)])

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
  def __init__(self, p):
    super().__init__()
    self.return_ff = p.return_ff 
    d_h = p.d_x
    # sublayer 1
    self.layernorm1 = nn.LayerNorm(d_h)
    self.selfAttn = SelfAttention(p)
    self.dropout1 = nn.Dropout(p.dropout)
    # sublayer 2
    self.layernorm2 = nn.LayerNorm(d_h)
    self.encAttn = SelfAttention(p)
    self.dropout2 = nn.Dropout(p.dropout)
    # sublayer 3
    self.layernorm3 = nn.LayerNorm(d_h)
    self.densefilter = PositionwiseFeedforward(hid_dim=d_h,
                                               pf_dim=p.d_f,
                                               dropout=p.dropout)
    self.dropout3 = nn.Dropout(p.dropout)

    # output
    self.layernorm4 = nn.LayerNorm(d_h)


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
      ea_reps['feedforward'] = ff.detach()
    reps = {'self_attention':sa_reps,
            'encoder_attention':ea_reps}

    return self.layernorm4(trg), reps


class Seq2Seq(nn.Module):
  def __init__(self, p, embedding, encoder, decoder, pad_idx):
    super().__init__()

    self.embedding = embedding
    self.encoder = encoder
    self.decoder = decoder
    self.pad_idx = pad_idx
    
    self.return_embeddings = p.return_embeddings

  def make_masks(self, src, trg):
    # src = [batch_size, src_seq_size]
    # trg = [batch_size, trg_seq_size]

    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
    # trg_mask = [batch_size, 1, trg_seq_size, 1]
    trg_len = trg.shape[1]

    if getattr(torch, "bool") and torch.__version__ != "1.2.0" and torch.device('cuda') == trg.device:
      # bug in torch 1.3.0 needs this workaround
      trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool, device=trg.device))
      trg_mask = trg_pad_mask & trg_sub_mask
    else:
      # this is the correct code (torch 1.2.0 and torch 1.4.0?)
      # workarond for torch.tril() not currently supporting bool types
      trg_sub_mask = torch.tril(
        torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

    # for running on philly
    if True:  # os.getenv("PHILLY_JOB_ID"):
      trg_mask = trg_pad_mask & trg_sub_mask.bool()
    else:
      trg_mask = trg_pad_mask & trg_sub_mask  #.bool()        

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
    # out = [batch_size, trg_seq_size, d_x]

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

  def greedy_inference(model, src, sos_idx, eos_idx, max_length, device):
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    src_emb = model.embedding(src)

    # run encoder
    enc_src = model.encoder(src_emb, src_mask)
    trg = torch.ones(src.shape[0], 1).fill_(sos_idx).type_as(src).to(device)

    done = torch.zeros(src.shape[0]).type(torch.uint8).to(device)
    for _ in range(max_length):
      trg_emb = model.embedding(trg)
      trg_mask = model.make_trg_mask(trg)
      # run decoder
      output = model.decoder(src=enc_src, trg=trg_emb,
                             src_mask=src_mask, trg_mask=trg_mask)
      logits = model.embedding.transpose_forward(output)
      pred = torch.argmax(logits[:,[-1],:], dim=-1)
      trg = torch.cat([trg, pred], dim=1)

      eos_match = (pred.squeeze(1) == eos_idx)
      #done = done | eos_match
      done = done | eos_match.byte()

      if done.sum() == src.shape[0]:
        break

    return trg
