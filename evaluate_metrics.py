import argparse
import json
import nltk
#from nltk import word_tokenize, bleu_score, meteor_score
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk import word_tokenize
from nltk.translate import meteor, bleu
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
from evaluate import load
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def evaluate_metrics(args=None):
    json_path = '{}/{}/{}.jsonl'.format(args.output_dir, args.domain, args.filename)
    length = args.length
    data = []
    bleu_list = []
    rouge_list = []
    meteor_list = []
    hypothesis_list = []
    reference_list = []
    bleu = load("bleu")
    with open(json_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print('data : ', len(data))
    if length == -1:
        length = len(data)
    fn = SmoothingFunction().method4
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for d in tqdm(data[:length]):
        hypothesis = d['output_str'].replace('<s>', '').replace('</s>', '').replace('<mask>', '').replace('<edu>', '')
        reference = d['gtruth_tgt'].replace('<s>', '').replace('</s>', '').replace('<mask>', '').replace('<edu>', '')
        hypothesis_list.append(hypothesis)
        reference_list.append([reference])
        
        rouge_ = scorer.score(reference,hypothesis)['rougeL'].fmeasure
        rouge_list.append(rouge_)
        
        
        
        
        reference = [word_tokenize(reference)]
        hypothesis = word_tokenize(hypothesis)
        #print(hypothesis)
        
        bleu_= sentence_bleu(reference,hypothesis, smoothing_function=fn)
        bleu_list.append(bleu_)
        
        meteor_= meteor(reference,hypothesis)
        meteor_list.append(meteor_)
        
        if args.verbose:
            print('bleu : ',bleu_)
            print('rouge : ',rouge_)
            print('meteor : ',meteor_)
    Sentence_bleu = np.mean(bleu_list)
    Rouge = np.mean(rouge_list)
    Meteor = np.mean(meteor_list)
    results = bleu.compute(predictions=hypothesis_list, references=reference_list)
    Corpus_bleu = results['bleu']
    print(f'BLEU-4(Sentence) : {Sentence_bleu:.6f}')
    print(f'BLEU-4(Corpus) : {Corpus_bleu:.6f}')
    print(f'ROUGE-L : {Rouge:.6f}')
    print(f'METEOR : {Meteor:.6f}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='rst')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--length', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    evaluate_metrics(args=args)
