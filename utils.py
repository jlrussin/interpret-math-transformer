import torch

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update_vis_params(params,p):
  if hasattr(params,'return_attention'):
    p.return_attention = params.return_attention
  else:
    p.return_attention = False
  if hasattr(params,'return_norms'):
    p.return_norms = params.return_norms
  else:
    p.return_norms = False
  if hasattr(params,'return_Q'):
    p.return_Q = params.return_Q
  else:
    p.return_Q = False
  if hasattr(params,'return_K'):
    p.return_K = params.return_K
  else:
    p.return_K = False
  if hasattr(params,'return_V'):
    p.return_V = params.return_V
  else:
    p.return_V = False
  if hasattr(params,'return_R'):
    p.return_R = params.return_R
  else:
    p.return_R = False
  if hasattr(params,'return_vbar'):
    p.return_vbar = params.return_vbar
  else:
    p.return_vbar = False
  if hasattr(params,'return_newv'):
    p.return_newv = params.return_newv
  else:
    p.return_newv = False
  if hasattr(params,'return_ff'):
    p.return_ff = params.return_ff
  else:
    p.return_ff = False
  if hasattr(params,'return_mha_res'):
    p.return_mha_res = params.return_mha_res
  else:
    p.return_mha_res = False
  if hasattr(params,'return_ff_res'):
    p.return_ff_res = params.return_ff_res
  else:
    p.return_ff_res = False
  if hasattr(params,'return_embeddings'):
    p.return_embeddings = params.return_embeddings 
  else:
    p.return_embeddings = False

class HyperParams:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def keys(self):
    return self.__dict__.keys()

  def __repr__(self):
    keys = self.__dict__.keys()
    items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
    return "{}:\n\t{}".format(type(self).__name__, "\n\t".join(items))

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

matched_dict = {'custom_adddiv': {'diffs': ['diff_sums', 'diff_y'],
                                  'filts': ['plus', 'div']}, 
                'custom_muldiv': {'diffs': ['diff_mul_num', 'diff_mul_den', 'diff_div'],
                                  'filts': ['mul_num', 'mul_den', 'div']}, 
                'custom_addmul': {'diffs': ['diff_mul', 'diff_sum'],
                                  'filts': ['mul', 'plus']}, 
                'custom_addmuldiv': {'diffs': ['diff_mul', 'diff_sum', 'diff_div'],
                                     'filts': ['mul', 'plus', 'div']}, 
                'custom_adddivadd': {'diffs': ['diff_sum1', 'diff_div', 'diff_sum2'],
                                     'filts': ['plus1', 'div', 'plus2']}, 
                'custom_adddiv2': {'diffs': ['diff_term1_sum', 'diff_term1_div', 
                                             'diff_term2_sum', 'diff_term2_div',
                                             'diff_sum'],
                                   'filts': ['term1_plus', 'term1_div', 
                                             'term2_plus', 'term2_div', 
                                             'plus']}}

constituent_dict = {}
constituent_dict['custom_adddiv'] = {'diff_sums': ['p2', 'p3', 'p4', 'p5', 'p6'],
                                     'diff_y': ['p2', 'p3', 'p4', 'p5', 'p6', 'p8', 'p9']} 
constituent_dict['custom_muldiv'] = {'diff_mul_num': ['p1', 'p2', 'p3'],                   
                                     'diff_mul_den': ['p8', 'p9', 'p10'],
                                     'diff_div': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11']}
constituent_dict['custom_addmul'] = {'diff_mul': ['p5', 'p6', 'p7'],
                                     'diff_sum': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']}
constituent_dict['custom_addmuldiv'] = {'diff_mul': ['p6', 'p7', 'p8'],
                                        'diff_sum': ['p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'],
                                        'diff_div': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11']}
constituent_dict['custom_adddivadd'] = {'diff_sum1': ['p2', 'p3', 'p4', 'p5', 'p6'],
                                        'diff_div': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9'],
                                        'diff_sum2': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13']}
constituent_dict['custom_adddiv2'] = {'diff_term1_sum': ['p2', 'p3', 'p4', 'p5', 'p6'],
                                      'diff_term1_div': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9'],
                                      'diff_term2_sum': ['p14', 'p15', 'p16', 'p17', 'p18'],
                                      'diff_term2_div': ['p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21'],
                                      'diff_sum': ['p{}'.format(p_i) for p_i in range(1,22)]}

def determine_op_match(custom_name, diff_name, filt_name):
  diffs = matched_dict[custom_name]['diffs']
  filts = matched_dict[custom_name]['filts']
  if diff_name in diffs and filt_name in filts:
    include = True 
    matched = diffs.index(diff_name) == filts.index(filt_name)
  else:
    include = False 
    matched = None 
  return include, matched

def determine_x_match(custom_name, diff_name, filt_name):
  diffs = ['diff_x{}'.format(i) for i in range(1,7)]
  filts = ['x{}'.format(i) for i in range(1,7)]
  if diff_name in diffs and filt_name in filts:
    include = True 
    matched = diffs.index(diff_name) == filts.index(filt_name)
  else:
    include = False 
    matched = None 
  return include, matched

def determine_con_match(custom_name, diff_name, filt_name):
  c_dict = constituent_dict[custom_name]
  all_filts = set([f for filt_list in c_dict.values() for f in filt_list])
  if diff_name in c_dict:
    if filt_name in c_dict[diff_name]:
      include = True
      matched = True 
    elif filt_name in all_filts:
      include = True
      matched = False
    else:
      include = False
      matched = None
  else:
    include = False 
    matched = None
  return include, matched

