import os
from collections import Counter

from torchtext.data import Field, Iterator
from torchtext.datasets import TranslationDataset

# TRAIN_FILE_NAME = "train"
# EVAL_FILE_NAME = "interpolate"

# INPUTS_FILE_ENDING = ".x"
# TARGETS_FILE_ENDING = ".y"

class DataLoader:
  def __init__(self, module_name, train_bs, eval_bs, device, vocab=None, 
    base_folder=None, train_name=None, eval_name=None, x_ext=None, y_ext=None, 
    tokens=None, specials=None, tokenizer=None, sort_within_batch=None, shuffle=None):

    self.module_name = module_name

    # split_chars = lambda x: list("".join(x.split()))
    split_chars = lambda x: list(x)  # keeps whitespaces

    if not tokenizer:
        tokenizer = split_chars

    # NOTE: on Jul-20-2020, removed fix_length=200 since it forces 
    # all batches to be of size (batch_size, 200) which
    # really wastes GPU memory
    source = Field(tokenize=tokenizer,
                   init_token='<sos>',
                   eos_token='<eos>',
                   batch_first=True)

    target = Field(tokenize=tokenizer,
                   init_token='<sos>',
                   eos_token='<eos>',
                   batch_first=True)

    base_folder = os.path.expanduser(base_folder)

    folder = os.path.join(base_folder, module_name)
    
    # fix slashes 
    folder = os.path.abspath(folder)

    print("loading FULL datasets from folder={}".format(folder))
    
    train_dataset, eval_dataset, _ = TranslationDataset.splits(
      path=folder,
      root=folder,
      exts=(x_ext, y_ext),
      fields=(source, target),
      train=train_name,
      validation=eval_name,
      test=eval_name)

    if vocab:
        print("Setting vocab to prebuilt file...")
        source.vocab = vocab
        target.vocab = vocab
    elif tokens:
        print("Building vocab from tokens...")
        #source.build_vocab(tokens, specials)
        counter = Counter(tokens)
        source.vocab = source.vocab_cls(counter, specials=specials)
        target.vocab = source.vocab
    else:
        print("Building vocab from TRAIN and EVAL datasets...")
        source.build_vocab(train_dataset, eval_dataset)
        target.vocab = source.vocab

    print("Creating iterators ...")
    do_shuffle = True if shuffle is None else shuffle
    train_iterator = Iterator(dataset=train_dataset,
                              batch_size=train_bs,
                              train=True,
                              repeat=True,
                              shuffle=do_shuffle,
                              sort_within_batch=sort_within_batch,
                              device=device)

    eval_iterator = Iterator(dataset=eval_dataset,
                             batch_size=eval_bs,
                             train=False,
                             repeat=False,
                             shuffle=False,
                             sort_within_batch=sort_within_batch,
                             device=device)

    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    
    self.train_iterator = train_iterator
    self.eval_iterator = eval_iterator

    self.source = source
    self.target = target
 
  def encode(self, str_list):
    return self.source.process(str_list)

  def decode(self, batch, remove_pad=False):
    itos = self.source.vocab.itos.copy()
    if remove_pad:
      itos[1] = ""
    #str_list = ["".join([itos[idx] for idx in row]) for row in batch.tolist()]
    str_list = ["".join([itos[idx] for idx in row]) for row in batch]
    return str_list


