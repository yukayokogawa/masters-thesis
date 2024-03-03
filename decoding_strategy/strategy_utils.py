import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def remove_nan(logits, filter_value=0.0):
    nan_indices = torch.isnan(logits)
    logits[nan_indices] = filter_value
    return logits


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1, probs=False):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if not probs:
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
    else:
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            values = torch.topk(logits, top_k)[0]
            batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
            
            logits = torch.where(logits < batch_mins,torch.ones_like(logits) * 0.0, logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = 0.0
        return logits



def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


def calc_banned_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    # Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens

def tag_kp_tokens_in_paragraph(tokens, kp_set):
    """Find tokens that belong to any KP from kp_set"""

    kp_tag = [0 for _ in tokens]
    first_token_to_kp = dict()
    for kp in kp_set:
        if len(kp) == 0: continue
        first = kp[0]
        if not first in first_token_to_kp:
            first_token_to_kp[first] = []
        first_token_to_kp[first].append(kp)

    ix = 0
    kp_start_idx = []

    while ix < len(tokens):
        cur_tok = tokens[ix]
        if not cur_tok in first_token_to_kp:
            ix += 1
            continue

        # find the longest matched sequence of tokens
        possible_kps = first_token_to_kp[cur_tok]
        longest_match = []
        for p in possible_kps:
            if len(p) + ix >= len(tokens):
                p = p[:len(tokens) - ix]
            if tokens[ix: len(p) + ix] == p \
                and len(p) > len(longest_match):
                longest_match = p

        if len(longest_match) > 0:
            kp_tag[ix: ix + len(longest_match)] = [1] * len(longest_match)
            kp_start_idx.append(ix)
            ix += len(longest_match)

        else:
            ix += 1
    return kp_tag, kp_start_idx

def get_edu_indices(input_ids, edu_idx=50265):
    batch_size = input_ids.size(0)
    edu_indices = []
    for b in range(batch_size):
        ids = input_ids[b]
        edu_pos = ids.eq(edu_idx)
        edu_index = edu_pos.sum().item()
        if edu_index > 3:
            edu_index = -1
        edu_indices.append(edu_index)
    return edu_indices

def get_edu_relations(all_edu_relations, indices):
    result = []
    for b, edu_index in enumerate(indices):
        if edu_index == -1:
            edu_index = 3
        result.append(all_edu_relations[b][edu_index])
    return result



def gather_nd(x, indices):
    newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([x.__getitem__(tuple(i)) for i in indices]).reshape(newshape)
    return out



def get_all_rst_labels(all_node_relations, max_node_num=3):
    batch_size = len(all_node_relations)
    all_rst_labels = []
    for b in range(batch_size):
        rst_labels = []
        for i in range(max_node_num):
            relation_label = all_node_relations[b][i]['relation_label']
            rst_labels.append(relation_label)
        rst_labels = torch.LongTensor(rst_labels).cuda()
        all_rst_labels.append(rst_labels)
    all_rst_labels = torch.stack(all_rst_labels, dim=0).cuda()
    return all_rst_labels

def get_prob_from_template(template, edu_reader, classifier, all_node_relations, edu_idx=50265, d_model=768, max_edu_num=4, max_node_num=3):
    batch_size = template.size(0)
    #print(template)
    outputs = edu_reader(
        input_ids=template,
    )
    hidden_states = outputs['decoder_hidden_states'][-1]
    
    edu_pos = template.eq(edu_idx)
    edu_indices = edu_pos.nonzero()
    
    edu_hidden_states = gather_nd(hidden_states, edu_indices)
    edu_hidden_states = edu_hidden_states.reshape([batch_size, max_edu_num, d_model])
    assert edu_hidden_states.size(1) == 4
    
    node_hidden_states = [[] for _ in range(max_node_num)]
    
    for b in range(batch_size):
        edu_hidden = edu_hidden_states[b]
        for i in range(max_node_num):
            node_relation = all_node_relations[b//2][i]
            rst_indices = node_relation['rst_indices'].cuda()
            node_hidden_states[i].append(edu_hidden.index_select(0,rst_indices))
    node_hidden_states = list(map(lambda x:torch.stack(x, dim=0), node_hidden_states))
    
    relation0 = classifier(node_hidden_states[0][:,0], node_hidden_states[0][:,1])
    relation1 = classifier(node_hidden_states[1][:,0], node_hidden_states[1][:,1])
    relation2 = classifier(node_hidden_states[2][:,0], node_hidden_states[2][:,1])
    
    
    predicted_relations = torch.stack(
        [relation0, relation1, relation2],
        dim=1,
    )
    predicted_probs = F.softmax(predicted_relations, dim=-1)
    
    all_rst_labels = get_all_rst_labels(
        all_node_relations,
    )
    all_probs = []
    for b in range(batch_size):
        probs = 0.0
        for i in range(max_node_num):
            correct_label = all_rst_labels[b//2,i].item()
            prob = predicted_probs[b, i, correct_label].item()
            probs += prob
        probs /= max_node_num
        all_probs.append(probs)
    return all_probs


    
def update_past(original_past, updated_past,no_template_indices):
    def update(x, new_x):
        result_x = x.clone()
        result_x[no_template_indices] = new_x
        return result_x
    result_past_key_values = []
    for n_layer, (orig_past, new_past) in enumerate(zip(original_past, updated_past)):
        result_past_kv = tuple(map(lambda x: update(x[0], x[1]), zip(orig_past, new_past)))
        result_past_key_values.append(result_past_kv)
    result_past_key_values = tuple(result_past_key_values)
    #print('result_past_key_values',result_past_key_values)
    return result_past_key_values




def select_indices_past(past_key_values, indices):
    def select(x):
        return x.index_select(0, indices)
    result_past_key_values = []
    for n_layer, past_kv in enumerate(past_key_values):
        result_past_kv = tuple(map(lambda x:select(x), past_kv))
        result_past_key_values.append(result_past_kv)
    result_past_key_values = tuple(result_past_key_values)
    return result_past_key_values
        


def bart_past_for_pplm(past_key_values):
    past = []
    enc_past = []
    for n_layer, _past in enumerate(past_key_values):
        key, value, enc_key, enc_value = _past
        key_value = torch.stack(
            [key, value],
            dim=0,
        )
        past.append(key_value)
        encoder_key_value = torch.stack(
            [enc_key, enc_value],
            dim=0,
        )
        enc_past.append(encoder_key_value)
    return past, enc_past



def pplm_past_for_bart(past, encoder_past):
    past_key_values = []
    for n_layer, (p, enc_p) in enumerate(zip(past, encoder_past)):
        key, value = p[0], p[1]
        enc_key, enc_value = enc_p[0], enc_p[1]
        _past_key_values = (key, value, enc_key, enc_value)
        past_key_values.append(_past_key_values)
    past_key_values = tuple(past_key_values)
    return past_key_values


def get_nucleus_satellite_hidden_states(
    generated_hidden_states,
    edu_hidden_states,
    nuclearity,
):
    nucleus_hidden_states = []
    satellite_hidden_states = []
    batch_size = generated_hidden_states.size(0)
    for b in range(batch_size):
        global_or_local = 'global' if b % 2 == 0 else 'local'
        original_b = b // 2
        generated_hidden = generated_hidden_states[b]
        edu_hidden = edu_hidden_states[original_b]
        global_nuc, local_nuc = nuclearity[original_b]
        if global_or_local == 'global' : 
            nuc = global_nuc
            index = 0
        else:
            nuc = local_nuc
            index = 1
        if nuc == 'nucleus':
            nucleus_hidden_states.append(generated_hidden)
            satellite_hidden_states.append(edu_hidden[index].unsqueeze(0))
        else:
            nucleus_hidden_states.append(edu_hidden[index].unsqueeze(0))
            satellite_hidden_states.append(generated_hidden)
    nucleus_hidden_states = torch.stack(nucleus_hidden_states, dim=0).cuda()
    satellite_hidden_states = torch.stack(satellite_hidden_states, dim=0).cuda()
    return nucleus_hidden_states, satellite_hidden_states
   
        
             
            
                        


    
    
