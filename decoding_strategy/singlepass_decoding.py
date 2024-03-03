import torch
import torch.nn.functional as F
from .strategy_utils import (
    remove_nan,
    top_k_top_p_filtering,
    RepetitionPenaltyLogitsProcessor,
)


import utils

from decoding_strategy import BaseDecoding

SMALL_CONST = 1e-15

class SinglePassDecoding(BaseDecoding):
    

    def generate(self, model, batch):
        #print(batch)
        net_input = utils.move_to_cuda(batch['net_input'])
        input_ids = net_input['input_ids']
        attention_mask = net_input['attention_mask']
        
        encoder = model.encoder.cuda()
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        decoder = model.model.cuda()
        outputs = decoder.generate(
            None,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            do_sample=True, 
            num_beams=1,
            repetition_penalty=1.0,
            max_length=self.max_tgt_len, 
            top_p=self.topp, 
            temperature=self.temperature, 
            decoder_start_token_id=0,
        )
        
        
        return outputs

