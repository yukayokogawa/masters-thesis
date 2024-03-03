"""Refinement"""
import json
import torch
from tqdm import tqdm
from sklearn import preprocessing as sklp
import utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
)
from bilinear_classification_head import BilinearClassificationHead

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
seed = 1
np.random.seed(seed)
    # Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
torch.backends.cudnn.benchmark = False
class EvaluationDataset(Dataset):

    def __init__(self, args, tokenizer, data_key='unpert', **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.data_key = data_key
        path = '/workspace/new_all_edu_results.jsonl'
        
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
            cur_obj = json.loads(ln)
            edus = cur_obj[self.data_key]
            if len(edus) > self.max_edu_num:
                continue

            cur_id = cur_obj['index']
            self.ID.append(cur_id)
            
            tree_str = cur_obj['tree']
            #print(tree_str)
            assigned_edu_index = utils.assign_tree_to_edu(tree_str)
            dict_edus = {}
            for index in range(self.max_edu_num):
                dict_edus[str(index)] = {
                    'edu_str' : '<edu>',
                }
            
            
            #print(edus)
            for index, edu in enumerate(edus):
                edu_index = assigned_edu_index[str(index)]
                edu_str = edu
                dict_edus[str(edu_index)]['edu_str'] = edu_str + '<edu>'
            #print(dict_edus)
            src_tokens = []
            for edu_index in range(self.max_edu_num):
                edu_index = str(edu_index)
                edu_str = dict_edus[edu_index]['edu_str']
                edu_tokens = self.tokenizer.encode(edu_str, add_special_tokens=False)
                src_tokens.extend(edu_tokens)
            src_tokens = [self.bos_idx] + src_tokens + [self.eos_idx]
            
            self.source.append(src_tokens)
            
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
            labels = self.rst_rel_labeler.transform(labels)
            self.target.append(labels)
            node0_label, node1_label, node2_label = labels
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
        
        encoder_input = torch.LongTensor(src_ids)
        
        tgt_ids = self.target[index]
        tgt_ids = torch.LongTensor(tgt_ids)

        
        edu_relations = self.all_edu_relations[index]
        
        node_relations = self.all_node_relations[index]

        

        ret_obj = dict(
            id=cur_id,
            encoder_input=encoder_input,
            #src_len=src_len,
            tgt_ids=tgt_ids,
            #template_ids=template_ids,
            #tgt_str=tgt_str,
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
            #src_len=[s['src_len'] for s in samples],
            #tgt_str=[s['tgt_str'] for s in samples],
            edu_relations=[s['edu_relations'] for s in samples],
            node_relations=[s['node_relations'] for s in samples],
        )
        
        labels = merge('tgt_ids')
        labels[labels == 0] = self.ignore_index
        ret_obj['labels'] = labels
        


        ret_obj['net_input'] = net_input


        return ret_obj
def get_dataloader(args, tokenizer, data_key='unpert'):
    test_dataset = EvaluationDataset(
        args=args,
        tokenizer=tokenizer,
        data_key=data_key
    )

    return DataLoader(test_dataset, batch_size=32,
                      collate_fn=test_dataset.collater)
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

def evaluation(args):
    data_key = args.data_key
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_tokens(['<edu>'])
    edu_idx = len(tokenizer)-1
    edu_reader = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-base',
        output_hidden_states=True,
    )
    edu_reader.resize_token_embeddings(len(tokenizer))
    edu_reader.eval()
    for param in edu_reader.parameters():
        param.requires_grad = False
    edu_reader = edu_reader.cuda()
    classifier = BilinearClassificationHead()
    classifier_path = 'classifier_head_bart_base_complex_latent2048_use_bilinear.pt'
    classifier_checkpoint = torch.load(classifier_path)
    classifier.load_state_dict(classifier_checkpoint)
    print('classifier load.')
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    classifier = classifier.cuda()

    test_dataloader = get_dataloader(args, tokenizer, data_key)

    corrects = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            net_input = utils.move_to_cuda(batch['net_input'])
            all_edu_relations = batch['edu_relations']
            all_node_relations = batch['node_relations']
            labels = batch['labels'].cuda()
            input_ids = net_input['input_ids']
            batch_size = input_ids.size(0)
            outputs = edu_reader(**net_input)
            hidden_states = outputs['decoder_hidden_states'][-1]
            hidden_size = hidden_states.size(-1)
            edu_pos = input_ids.eq(edu_idx)
            edu_indices = edu_pos.nonzero()
    
            edu_hidden_states = gather_nd(hidden_states, edu_indices)
            edu_hidden_states = edu_hidden_states.reshape([batch_size, 4, hidden_size])
            node_hidden_states = [[] for _ in range(3)]
    
            for b in range(batch_size):
                edu_hidden = edu_hidden_states[b]
                for i in range(3):
                    node_relation = all_node_relations[b][i]
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
            predictions = torch.argmax(predicted_relations, dim=-1)
            preds_ = predictions.view(-1)
            labels_ = labels.view(-1)
            labels = labels_[labels_.ne(-100)]
            preds = preds_[labels_.ne(-100)]
    
            corr = preds.eq(labels)
            accuracy = sum(corr).item() / len(corr)
            
            print(f'Accuracy : {accuracy:.6f}')
            correct = corr.cpu().tolist()
            corrects.extend(correct)
    accuracy = np.mean(corrects)
    print(f'Result(Accuracy) : {accuracy:.6f}')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-key', type=str, default='unpert')
    args = parser.parse_args()
    evaluation(args)