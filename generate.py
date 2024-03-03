import time
import json
import os
import logging

from tqdm import tqdm
import torch
import torch.nn as nn

from system import BARTSeq2seq
from options import get_generation_parser
import utils

import random
import numpy as np

from decoding_strategy import (
    SinglePassDecoding,
    PPLMDecoding,
)

from transformers import BartForConditionalGeneration

from bilinear_classification_head import BilinearClassificationHead

logger = logging.getLogger(__name__)

MASK_TOK = '<mask>'
BOS_TOK = '<s>'
BOS_IDX = 0
PAD_IDX = 1
EOS_IDX = 2
MASK_IDX = 50264

from evaluate import load
import numpy as np

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")
bertscore = load("bertscore")



def generate():
    parser = get_generation_parser()
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False
    
    
    ckpt_path = utils.get_latest_ckpt_path(args.ckpt_dir)
    print(f'Evaluating on {ckpt_path}')

    t0 = time.time()
    checkpoint = torch.load(ckpt_path)
    print('Checkpoint loading time: {:.2f} secs'
          .format(time.time() - t0))

    if not os.path.exists(f'output/{args.domain}/'):
        os.makedirs(f'output/{args.domain}/')
    output_path = f'output/{args.domain}/{args.output_name}.jsonl'

    system = BARTSeq2seq(hparams=args, is_inference=True).cuda()
    test_dataloader = system.test_dataloader()
    system.load_state_dict(checkpoint['state_dict'])
    if args.fp16:
        system.model = system.model.half()

    if args.n_gpus > 1:
        system.model = nn.DataParallel(system.model)
    del checkpoint
    torch.cuda.empty_cache()

    fout = open(output_path, 'w')

    if args.use_pplm:
        edu_reader = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base',
            output_hidden_states=True,
            )
        edu_reader.resize_token_embeddings(len(system.tokenizer))
        edu_reader.eval()
        for param in edu_reader.parameters():
            param.requires_grad = False
        edu_reader = edu_reader.cuda()
        classifier = BilinearClassificationHead()
        classifier_path = 'classifier_head_bart_base_complex_latent2048_use_bilinear.pt'
        classifier_checkpoint = torch.load(classifier_path)
        classifier.load_state_dict(classifier_checkpoint)
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        classifier = classifier.cuda()
        del classifier_checkpoint 
        decoding_strategy = PPLMDecoding(args, system.tokenizer, edu_reader, classifier)
    else:
        decoding_strategy = SinglePassDecoding(args, system.tokenizer)
    
    reference_list = []
    hypothesis_list = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            net_inputs = utils.move_to_cuda(batch['net_input'])
            output_ids = decoding_strategy.generate(system.model,
                                                            batch)
            
            output_str = system.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for b, sample_id in enumerate(batch['id']):
                
                hypo_str = output_str[b]
                
                ret_obj = dict(
                    id=sample_id,
                    output_str=hypo_str,
                )
                
                if 'lm_labels' in batch:
                    cur_tgt_ids = batch['lm_labels'][b]
                    cur_tgt_ids = cur_tgt_ids[cur_tgt_ids >= 0]
                    cur_tgt_str = system.tokenizer.decode(cur_tgt_ids, skip_special_tokens=True).strip()
                    ret_obj['gtruth_tgt'] = cur_tgt_str
                
                hypothesis = hypo_str.replace('<edu>', '').lower()
                print('hypothesis',hypothesis)
                reference = cur_tgt_str.replace('<edu>', '').lower()
                hypothesis_list.append(hypothesis)
                reference_list.append([reference])
                
                if b%4 == 3:
                    bleu_results = bleu.compute(predictions=hypothesis_list, references=reference_list)
                    rouge_results = rouge.compute(predictions=hypothesis_list, references=reference_list)
                    meteor_results = meteor.compute(predictions=hypothesis_list, references=reference_list)
                    #bertscore_results = bertscore.compute(predictions=hypothesis_list, references=reference_list, lang='en')
                
                    BLEU = bleu_results['bleu']
                    ROUGE = rouge_results['rougeL']
                    METEOR = meteor_results['meteor']
                    #BERT_SCORE = np.mean(bertscore_results['f1'])
                
                    print(f'BLEU-4 : {BLEU:.6f}')
                    print(f'ROUGE-L : {ROUGE:.6f}')
                    print(f'METEOR : {METEOR:.6f}')
                    
                
    
                print('write')
                fout.write(json.dumps(ret_obj) + "\n")
    
    
    fout.close()

if __name__ == '__main__':
    generate()
