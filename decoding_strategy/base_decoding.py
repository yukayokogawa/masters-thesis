import torch
from .strategy_utils import (
    select_indices_past,
    update_past,
)

class BaseDecoding(object):

    def __init__(self, args, tokenizer, edu_reader=None, classifier=None):
        super().__init__()
        self.multi_gpus = True if args.n_gpus > 1 else False
        self.quiet = args.quiet
        self.args = args
        self.domain = args.domain
        self.tokenizer = tokenizer

        self.domain_to_max_len = {
            'arggen': 140,
            'opinion': 243,
            'news': 335
        }

        self.pad_idx = 1
        self.mask_idx = 50264
        self.eos_idx = 2
        self.decoder_bos_idx = 0
        self.edu_idx = 50265

        self.do_sampling = args.do_sampling
        self.topk = args.sampling_topk
        self.topp = args.sampling_topp
        self.temperature = args.temperature

        self.max_tgt_len = args.max_tgt_len
        
        self.use_pplm = args.use_pplm
        self.no_template = args.no_template
        if self.use_pplm:
            #self.no_perturb = args.no_perturb
            
            self.edu_reader = edu_reader
            self.classifier = classifier
            self.pplm_sample_times = 2
            self.max_edu_num = 4
            self.pplm_parameters = {
                'decay' : args.decay,
                'stepsize' : args.stepsize,
                'gm_scale' : args.gm_scale,
                'kl_scale' : args.kl_scale,
                'gamma' : args.gamma,
                'verbosity_level' : args.verbosity_level,
                #'verbosity_level' : 3,
                'num_iterations' : args.num_iterations,
            }
            print(self.pplm_parameters)

    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids=None,
        past_key_values=None,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        return {
            'input_ids' : None,
            'decoder_input_ids' : decoder_input_ids,
            'past_key_values' : past_key_values,
            'encoder_outputs' : (encoder_hidden_states,),
            'attention_mask' : attention_mask,
        }
    
    
    def generate(self, **kwargs):
        raise NotImplementedError