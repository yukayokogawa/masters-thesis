from transformers import (
    BartConfig,
    BartForConditionalGeneration,
)

import torch
import torch.nn as nn

class GeneratorModel(nn.Module):
    def __init__(self, vocab_size=50265, is_inference=False):
        super().__init__()
        self.use_cache = is_inference
        self.config = BartConfig.from_pretrained(
            'facebook/bart-base',
            return_dict=True,
            output_hidden_states=True,
            use_cache=self.use_cache,
        )
        
        self.model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base',
            config=self.config,
        )
        
        self.vocab_size = vocab_size
        
        self.config.vocab_size = self.vocab_size
        
        self.model.resize_token_embeddings(self.vocab_size)
        
        self.encoder = self.model.get_encoder()
        
        self.decoder = self.model.get_decoder()
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        outputs = self.model(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )
        return {
            'outputs' : outputs,
            'encoder_outputs' : encoder_outputs,
        }