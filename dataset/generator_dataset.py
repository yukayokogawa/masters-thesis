"""Refinement"""
import json
import torch
from tqdm import tqdm
from sklearn import preprocessing as sklp
import utils

import numpy as np

rst_rel_li = [
    "<pad>",
    "Elaboration",
    "Attribution",
    "Joint",
    "Same-unit",
    "Contrast",
    "Explanation",
    "Background",
    "Cause",
    "Enablement",
    "Evaluation",
    "Temporal",
    "Condition",
    "Comparison",
    "Topic-Change",
    "Summary",
    "Manner-Means",
    "Textual-organization",
    "Topic-Comment",
]

class GeneratorDataset(object):

    def __init__(self, args, set_type='train', tokenizer=None, is_inference=False, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.set_type = set_type
        path = f'../data_generator_height2/generator_{self.set_type}.jsonl'
        
        self.rst_rel_labeler = sklp.LabelEncoder()
        self.rst_rel_labeler.fit(rst_rel_li)
        
        self.sep_tok = '<s>'
        self.sep_idx = 0

        self.bok_tok = '<s>'
        self.bok_idx = 0

        self.bos_tok = '<s>'
        self.bos_idx = 0

        self.pad_tok = '<pad>'
        self.pad_idx = 1

        self.mask_tok = '<mask>'
        self.mask_idx = 50264

        self.eos_tok = '</s>'
        self.eos_idx = 2
        
        self.edu_tok = '<edu>'
        self.edu_idx = 50265
        
        self.ignore_index = -100
        
        self.max_edu_num = 4

        self.ID = []
        self.source = []
        self.target = []
        self.template = []
        self.target_str = []
        self.all_edu_relations = []
        self.all_node_relations = []
        
        
        self.random = np.random.RandomState(42)
        
        self.load_raw_dataset(path=path)
    
    def __len__(self):
        return len(self.ID)

    def load_raw_dataset(self, path):
        """Load raw data for refinement dataset"""

        print(f'loading {path}')
        for idx, ln in enumerate(tqdm(open(path))):
            """
            if idx < 8:
                continue
            """
            cur_obj = json.loads(ln)
            
            tree_str = cur_obj['tree']
            assigned_edu_index = utils.assign_tree_to_edu(tree_str)
            dict_edus = {}
            for index in range(self.max_edu_num):
                dict_edus[str(index)] = {
                    'edu_str' : '',
                    'edu_tokens' : [],
                    'template' : []
                }
            
            edus = cur_obj['edus']
            
            for index, edu in enumerate(edus):
                edu_index = assigned_edu_index[str(index)]
                edu_str = edu['edu_str']
                dict_edus[str(edu_index)]['edu_str'] = edu_str
                
                edu_tokens = edu['edu_tokens']
                dict_edus[str(edu_index)]['edu_tokens'] = edu_tokens
                
                edu_template = edu['template']
                edu_template = [template_token if template_token is not None else self.mask_idx for template_token in edu_template]
                dict_edus[str(edu_index)]['template'] = edu_template
            #print(dict_edus)
            tgt_tokens = []
            template_tokens = []
            tgt_str = ''
            for edu_index in range(self.max_edu_num):
                edu_index = str(edu_index)
                edu_str = dict_edus[edu_index]['edu_str']
                tgt_str += edu_str
                edu_tokens = dict_edus[edu_index]['edu_tokens']
                edu_tokens = edu_tokens + [self.edu_idx]
                tgt_tokens.extend(edu_tokens)
                edu_template = dict_edus[edu_index]['template']
                edu_template = edu_template + [self.edu_idx]
                template_tokens.extend(edu_template)
            tgt_tokens = [self.bos_idx] + tgt_tokens + [self.eos_idx]
            template_tokens = [self.mask_idx] + template_tokens + [self.mask_idx]
            self.target.append(tgt_tokens)
            self.template.append(template_tokens)
 
            self.target_str.append(tgt_str)
            

            cur_src_ids = template_tokens
            self.source.append(cur_src_ids)
            self.ID.append(cur_obj['id'])
            labels = []
            rst_relations = cur_obj['rst_relations']
            node0_label = rst_relations[0]['relation_label']
            labels.append(node0_label)
            node0_nuc0, node0_nuc1 = rst_relations[0]['nuclearity'].split('-')
            node1_label = rst_relations[1]['relation_label']
            labels.append(node1_label)
            node1_nuc = rst_relations[1]['nuclearity']
            if node1_nuc == '<pad>':
                node1_nuc0, node1_nuc1 = 'satellite', 'nucleus'
            else:
                node1_nuc0, node1_nuc1 = node1_nuc.split('-')
            node2_label = rst_relations[2]['relation_label']
            labels.append(node2_label)
            node0_label, node1_label, node2_label = self.rst_rel_labeler.transform(labels)
            node2_nuc = rst_relations[2]['nuclearity']
            if node2_nuc == '<pad>':
                node2_nuc0, node2_nuc1 = 'satellite', 'nucleus'
            else:
                node2_nuc0, node2_nuc1 = node2_nuc.split('-')
            edu_relations = []
            
            edu0 = {
                'index' : 3,
                'path' : [0,0],
                'relation_label' : [node0_label, node1_label],
                'nuclearity' : [node0_nuc0, node1_nuc0],
                'rst_indices' : torch.LongTensor([3,1]),
            }
            edu_relations.append(edu0)
            
            
            edu1 = {
                'index' : 4,
                'path' : [0,1],
                'relation_label' : [node0_label, node1_label],
                'nuclearity' : [node0_nuc0, node1_nuc1],
                'rst_indices' : torch.LongTensor([3,0]),
            }
            edu_relations.append(edu1)
            
            
            edu2 = {
                'index' : 5,
                'path' : [1,0],
                'relation_label' : [node0_label, node2_label],
                'nuclearity' : [node0_nuc1, node2_nuc0],
                'rst_indices' : torch.LongTensor([1,3]),
            }
            edu_relations.append(edu2)
            
            edu3 = {
                'index' : 6,
                'path' : [1,1],
                'relation_label' : [node0_label, node2_label],
                'nuclearity' : [node0_nuc1, node2_nuc1],
                'rst_indices' : torch.LongTensor([1,2]),
            }
            edu_relations.append(edu3)
                
            self.all_edu_relations.append(edu_relations)
            
            node_relations = []
            if node0_nuc0 == 'nucleus':
                rst_indices = torch.LongTensor([1,3])
            else:
                rst_indices = torch.LongTensor([3,1])
            
            node0 = {
                'index' : 0,
                'path' : [],
                'relation_label' : node0_label,
                'nuclearity' : [node0_nuc0, node0_nuc1],
                'rst_indices' : rst_indices,
            }
            node_relations.append(node0)
            if node1_nuc0 == 'nucleus':
                rst_indices = torch.LongTensor([0,1])
            else:
                rst_indices = torch.LongTensor([1,0])
            node1 = {
                'index' : 1,
                'path' : [0],
                'relation_label' : node1_label,
                'nuclearity' : [node1_nuc0, node1_nuc1],
                'rst_indices' : rst_indices,
            }
            node_relations.append(node1)
            if node2_nuc0 == 'nucleus':
                rst_indices = torch.LongTensor([2,3])
            else:
                rst_indices = torch.LongTensor([3,2])
            node2 = {
                'index' : 2,
                'path' : [1],
                'relation_label' : node2_label,
                'nuclearity' : [node2_nuc0, node2_nuc1],
                'rst_indices' : rst_indices,
            }
            node_relations.append(node2)
            
            self.all_node_relations.append(node_relations)
            
            

            

    def __getitem__(self, index):
        
        cur_id = self.ID[index]
        src_ids = self.source[index]
        src_len = len(src_ids)
        encoder_input = torch.LongTensor(src_ids)
        
        tgt_ids = self.target[index]
        tgt_ids = torch.LongTensor(tgt_ids)

        template_ids = self.template[index]
        template_ids = torch.LongTensor(template_ids)
        
        tgt_str = self.target_str[index]
        
        edu_relations = self.all_edu_relations[index]
        
        node_relations = self.all_node_relations[index]

        

        ret_obj = dict(
            id=cur_id,
            encoder_input=encoder_input,
            src_len=src_len,
            tgt_ids=tgt_ids,
            template_ids=template_ids,
            tgt_str=tgt_str,
            edu_relations=edu_relations,
            node_relations=node_relations,
        )

        
        return ret_obj
    
    def collater(self, samples):
        def merge(key, is_list=False, pad_idx=self.pad_idx):
            if is_list:
                res = []
                for i in range(len(samples[0][key])):
                    res.append(utils.collate_tokens(
                        [s[key][i] for s in samples], pad_idx=pad_idx,
                    ))
                return res
            else:
                return utils.collate_tokens(
                    [s[key] for s in samples], pad_idx=pad_idx,
                )

        input_ids = merge('encoder_input')
        attention_mask = input_ids.ne(1).long()

        net_input = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        ret_obj = dict(
            id=[s['id'] for s in samples],
            src_len=[s['src_len'] for s in samples],
            tgt_str=[s['tgt_str'] for s in samples],
            edu_relations=[s['edu_relations'] for s in samples],
            node_relations=[s['node_relations'] for s in samples],
        )
        
        tgt_ids = merge('tgt_ids')
        decoder_input_ids = tgt_ids[:, :-1].contiguous()
        decoder_input_ids[decoder_input_ids == self.eos_idx] = self.pad_idx
        net_input['decoder_input_ids'] = decoder_input_ids
        lm_labels = tgt_ids[:, 1:].clone()
        lm_labels[lm_labels == self.pad_idx] = -100
        ret_obj['lm_labels'] = lm_labels
        
        template_ids = merge('template_ids')
        template_ids = template_ids[:, 1:].clone()
        ret_obj['template'] = template_ids


        ret_obj['net_input'] = net_input


        return ret_obj