import torch
import torch.nn.functional as F
from .strategy_utils import (
    remove_nan,
    top_k_top_p_filtering,
    get_edu_indices,
    gather_nd,
    select_indices_past,
    update_past,
    get_prob_from_template,
    get_edu_relations,
    RepetitionPenaltyLogitsProcessor,
    )

from .pplm_utils import (
    perturb_past,
)
"""
from .pplm import (
    perturb_past,
)
"""
import utils

from decoding_strategy import BaseDecoding

SMALL_CONST = 1e-15



class PPLMDecoding(BaseDecoding):
    def prepare_inputs_for_perturb(
        self,
        decoder_input_ids=None,
        no_template_indices=None,
        next_token_logits=None,
        past_key_values=None,
        unpert_past_key_values=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        template=None,
        cur_edu_relations=None,
        all_edu_hidden_states=None,
        all_node_relations=None,
    ):
        pert_indices = no_template_indices.view(-1,1).repeat(1, self.pplm_sample_times).view(-1).cuda()
        
        decoder_input_ids = decoder_input_ids.index_select(0, pert_indices)
        unpert_logits = next_token_logits.index_select(0, pert_indices)
        past_key_values = select_indices_past(
            past_key_values=past_key_values,
            indices=pert_indices,
            )
        unpert_past_key_values = select_indices_past(
            past_key_values=unpert_past_key_values,
            indices=pert_indices,
            )
        encoder_hidden_states = encoder_hidden_states.index_select(0, pert_indices)
        encoder_attention_mask = encoder_attention_mask.index_select(0, pert_indices)
        
        template = template.index_select(0, pert_indices)
        
        relation_labels = []
        nuclearity = []
        rst_indices = []
        edu_hidden_states = []
        
        for index in no_template_indices:
            cur_edu_relation = cur_edu_relations[index]
            relation_label = cur_edu_relation['relation_label']
            relation_label = torch.LongTensor(relation_label).cuda()
            relation_labels.append(relation_label)
            nuclearity.append(cur_edu_relation['nuclearity'])
            _rst_indices = cur_edu_relation['rst_indices'].cuda()
            rst_indices.append(_rst_indices)
            _edu_hidden_states = all_edu_hidden_states[index]
            _edu_hidden_states = _edu_hidden_states.index_select(0, _rst_indices)
            edu_hidden_states.append(_edu_hidden_states)
 
        relation_labels = torch.stack(relation_labels, dim=0).cuda()
        edu_hidden_states = torch.stack(edu_hidden_states, dim=0).cuda()
        all_node_relations = [all_node_relations[index] for index in no_template_indices]
        return {
            'past_key_values' : past_key_values,
            'last' : decoder_input_ids[:, -1:],
            'unpert_logits' : unpert_logits,
            'unpert_past_key_values' : unpert_past_key_values,
            'encoder_hidden_states' : encoder_hidden_states,
            'attention_mask' : encoder_attention_mask,
            'template' : template,
            'relation_labels' : relation_labels,
            'nuclearity' : nuclearity,
            'edu_hidden_states' : edu_hidden_states,
            'all_node_relations' : all_node_relations,
        }
    
    def generate_text_pplm(
        self,
        model=None,
        edu_reader=None,
        classifier=None,
        cur_len=1,
        #pplm_inputs
        past_key_values=None,
        last=None,
        unpert_past_key_values=None,
        unpert_logits=None,
        encoder_hidden_states=None,
        attention_mask=None,
        template=None,
        relation_labels=None,
        nuclearity=None,
        edu_hidden_states=None,
        all_node_relations=None,
        #pplm_parameters
        stepsize=0.01,
        num_iterations=3,
        decay=False,
        gamma=1.5,
        gm_scale=0,
        kl_scale=0.01,
        verbosity_level=1,
    ):
        #stepsize = stepsize / cur_len
        expanded_batch_size = last.size(0)
        pert_past, _, _, _ = perturb_past(
            #pplm_inputs
            past_key_values=past_key_values,
            model=model,
            last=last,
            unpert_past_key_values=unpert_past_key_values,
            unpert_logits=unpert_logits,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            classifier=classifier,
            relation_labels=relation_labels,
            nuclearity=nuclearity,
            edu_hidden_states=edu_hidden_states,
            loss_type=2,
            #pplm_parameters
            decay=decay,
            gamma=gamma,
            num_iterations=num_iterations,
            stepsize=stepsize,
            kl_scale=kl_scale,
            device='cuda',
            verbosity_level=verbosity_level,
        )
        
        
        pert_model_inputs = self.prepare_inputs_for_generation(
            decoder_input_ids=last,
            past_key_values=pert_past,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        model_outputs = model(**pert_model_inputs)
        pert_model_outputs = model_outputs['outputs']
        pert_logits = pert_model_outputs['logits'][:, -1, :]
        pert_probs = F.softmax(pert_logits, dim=-1)
        unpert_probs = F.softmax(unpert_logits, dim=-1)
        
        pert_next_token_probs = ((pert_probs ** gm_scale) * (
                unpert_probs ** (1 - gm_scale)))  #+ SMALL_CONST
        
        pert_next_token_probs = top_k_top_p_filtering(
            pert_next_token_probs,
            top_k=self.topk,
            top_p=self.topp,
            probs=True,
        )#+ SMALL_CONST
        
        """
        if torch.sum(pert_next_token_probs) <= 1:
            pert_next_token_probs = pert_next_token_probs / torch.sum(pert_next_token_probs)
        """
        pert_next_token = torch.multinomial(pert_next_token_probs, num_samples=1)
        #print('pert_next_token',pert_next_token)
        pert_template = template.clone()
        pert_template[:, cur_len-1] = pert_next_token.squeeze(-1)
        
        pert_label_probs = get_prob_from_template(
            template=pert_template,
            edu_reader=edu_reader,
            classifier=classifier,
            all_node_relations=all_node_relations,
        )
        
        selected_indices = []
        batch_size = expanded_batch_size // 2
        for b in range(batch_size):
            global_b = 2*b
            local_b = 2*b + 1
            global_label_prob = pert_label_probs[global_b]
            local_label_prob = pert_label_probs[local_b]
            if global_label_prob > local_label_prob:
                selected_indices.append(global_b)
            else:
                selected_indices.append(local_b)
        
        
        selected_indices = torch.LongTensor(selected_indices).cuda()
        pert_logits = pert_logits.index_select(0, selected_indices)
        pert_next_token_probs = pert_next_token_probs.index_select(0, selected_indices)
        pert_next_token = pert_next_token.index_select(0, selected_indices)
        pert_past = select_indices_past(
            past_key_values=pert_past,
            indices=selected_indices,
        )
        
        return {
            'pert_logits' : pert_logits,
            'pert_next_token_probs' : pert_next_token_probs,
            'pert_next_token' : pert_next_token,
            'pert_past' : pert_past,
        }
    def update_outputs_from_perturb(
        self,
        no_template_indices=None,
        next_token_logits=None,
        next_token=None,
        next_token_probs=None,
        chosen_token_probs=None,
        past_key_values=None,
        pert_past=None,
        pert_logits=None,
        pert_next_token=None,
        pert_next_token_probs=None,
    ):
        next_token_logits[no_template_indices] = pert_logits
        next_token[no_template_indices] = pert_next_token
        next_token_probs[no_template_indices] = pert_next_token_probs
        
        for index in no_template_indices:
            chosen_token_id = next_token[index].item()
            chosen_token_prob = next_token_probs[index, chosen_token_id]
            chosen_token_probs[index] = chosen_token_prob
        
        past_key_values = update_past(
            original_past=past_key_values,
            updated_past=pert_past,
            no_template_indices=no_template_indices,
        )

        return {
            'past_key_values' : past_key_values,
            'next_token_logits' : next_token_logits,
            'next_token' : next_token,
            'next_token_probs' : next_token_probs,
            'chosen_token_probs' : chosen_token_probs,
        }
                
        
        
        
        
    
    
    

    def generate(self, model, batch):
        #print(batch)
        net_input = utils.move_to_cuda(batch['net_input'])
        encoder_input_ids = net_input['input_ids']
        encoder_attention_mask = net_input['attention_mask']
        batch_size = encoder_input_ids.size(0)
        
        template = batch['template'].cuda()
        template_lens = template.ne(1).sum(-1)
        all_edu_relations = batch['edu_relations']
        all_node_relations = batch['node_relations']
        max_template_len = max(template_lens) 
        max_tgt_len = self.max_tgt_len
        if self.multi_gpus:
            encoder = model.module.get_encoder()
        else:
            encoder = model.get_encoder()
            
        encoder_outputs = encoder(
            encoder_input_ids,
            attention_mask=encoder_attention_mask,
            )
        encoder_hidden_states = encoder_outputs['last_hidden_state']
        
        encoder_edu_pos = encoder_input_ids.eq(self.edu_idx)
        encoder_edu_indices = encoder_edu_pos.nonzero()
        edu_hidden_states = gather_nd(encoder_hidden_states, encoder_edu_indices)
        all_edu_hidden_states = edu_hidden_states.reshape([batch_size, self.max_edu_num, edu_hidden_states.size(-1)])
        
        # create empty decoder_input_ids
        input_ids = torch.full(
            (batch_size, 1),
            self.decoder_bos_idx,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        last = input_ids[:, -1:]
        cur_len = 1
        probs = [[] for _ in range(batch_size)]

        unfinished_sents = input_ids.new(batch_size).fill_(1)
        past_key_values = None
        
        #max_len = 300

        logits_processor = RepetitionPenaltyLogitsProcessor(1.01)

        while cur_len < max_tgt_len:
            cur_edu_indices = get_edu_indices(input_ids=input_ids)
            cur_edu_relations = get_edu_relations(
                all_edu_relations=all_edu_relations,
                indices=cur_edu_indices,
            )
            model_inputs = self.prepare_inputs_for_generation(
                decoder_input_ids=input_ids,
                #decoder_input_ids=last,
                #past_key_values=past_key_values,
                past_key_values=None,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            model_outputs = model(**model_inputs)
            outputs = model_outputs['outputs']
            next_token_logits = outputs['logits'][:, -1, :]
            unpert_past_key_values = outputs['past_key_values']
            last_hidden_state = outputs['decoder_hidden_states'][-1]
            #next_token_logits = logits_processor(input_ids, next_token_logits)
            #next_token_logits[cur_len <= template_lens, self.eos_idx] = -10000.
            if cur_len < max_template_len:
                cur_template = template[:, cur_len-1]
                #print('cur_template',cur_template)
                no_template_pos = cur_template.eq(self.mask_idx)
                no_template_pos = no_template_pos * (unfinished_sents)
                no_template_indices = no_template_pos.nonzero()
                no_template_indices = no_template_indices.view(-1)
                
                no_edu_pos = cur_template.ne(self.edu_idx)
                no_edu_pos = no_edu_pos * (unfinished_sents)
                no_edu_indices = no_edu_pos.nonzero()
                no_edu_indices = no_edu_indices.view(-1)
                next_token_logits[no_edu_indices, self.edu_idx] = -10000.
            
            
            if self.do_sampling:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if self.temperature != 1.0:
                    next_token_logits = next_token_logits / self.temperature
                # Top-p/top-k filtering
                #print(next_token_logits)
                next_token_logits = remove_nan(next_token_logits)
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
                # Sample
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                chosen_token_probs = next_token_probs.gather(1, next_token)
            else:
                # Greedy decoding
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_logits, dim=-1)
                chosen_token_probs = next_token_probs.gather(1, next_token)
            #"""
            if cur_len < max_template_len:
                for index in range(batch_size):
                    if not (index in no_edu_indices):
                        cur_template_id = cur_template[index].item()
                        cur_template_prob = next_token_probs[index, cur_template_id]
                        chosen_token_probs[index] = cur_template_prob
                        next_token[index] = cur_template[index]
            #"""
            if not self.no_template:
                for index in range(batch_size):
                    if not (index in no_template_indices):
                        cur_template_id = cur_template[index].item()
                        #print(self.tokenizer.convert_ids_to_tokens(cur_template_id))
                        if cur_template_id != self.edu_idx:
                            leftmost = max(0, cur_len - 10)
                            _history = input_ids[index][leftmost:].tolist()
    
                            if cur_template_id in _history:
                                continue
                         
                        cur_template_prob = next_token_probs[index, cur_template_id]
                        chosen_token_probs[index] = cur_template_prob
                        next_token[index] = cur_template[index]
            """
            else:
                no_template_pos = no_edu_pos
                no_template_indices = no_edu_indices
                for index in range(batch_size):
                    if not (index in no_template_indices):
                        cur_template_id = cur_template[index].item()
                        cur_template_prob = next_token_probs[index, cur_template_id]
                        chosen_token_probs[index] = cur_template_prob
                        next_token[index] = cur_template[index]
            """
            #if no_template_indices.size(0) == 0 or past_key_values is None:
            if no_edu_indices.size(0) == 0 or past_key_values is None:
                past_key_values = unpert_past_key_values
            else:
                pplm_inputs = self.prepare_inputs_for_perturb(
                    decoder_input_ids=input_ids,
                    #no_template_indices=no_template_indices,
                    no_template_indices=no_edu_indices,
                    next_token_logits=next_token_logits,
                    past_key_values=past_key_values,
                    unpert_past_key_values=unpert_past_key_values,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    template=template,
                    cur_edu_relations=cur_edu_relations,
                    all_edu_hidden_states=all_edu_hidden_states,
                    all_node_relations=all_node_relations,
                )
                pplm_outputs = self.generate_text_pplm(
                    model=model,
                    classifier=self.classifier,
                    edu_reader=self.edu_reader,
                    cur_len=cur_len,
                    **pplm_inputs,
                    **self.pplm_parameters,
                )
                updated_outputs = self.update_outputs_from_perturb(
                    #no_template_indices=no_template_indices,
                    no_template_indices=no_edu_indices,
                    next_token_logits=next_token_logits,
                    next_token=next_token,
                    next_token_probs=next_token_probs,
                    chosen_token_probs=chosen_token_probs,
                    past_key_values=past_key_values,
                    **pplm_outputs,
                )
                #next_token_logits = updated_outputs['next_token_logits']
                next_token = updated_outputs['next_token']
                #next_token_probs = updated_outputs['next_token_probs']
                chosen_token_probs = updated_outputs['chosen_token_probs']
                past_key_values = updated_outputs['past_key_values']
 
            #chosen_token_probs = next_token_probs.gather(1, next_token.view(-1, 1))
            for b in range(batch_size):
                probs[b].append(chosen_token_probs[b, 0].item())

            # pad finished sentences if eos_token_id exist
            #"""
            if cur_len < max_template_len:
                if not self.no_template:
                    tokens_to_add = next_token.squeeze() * no_template_pos + cur_template *  (1 - no_template_pos)
                    tokens_to_add = tokens_to_add * unfinished_sents + (self.pad_idx) * (1 - unfinished_sents)
                else:
                    tokens_to_add = next_token.squeeze() * no_edu_pos + cur_template *  (1 - no_edu_pos)
                    tokens_to_add = tokens_to_add * unfinished_sents + (self.pad_idx) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token.squeeze() * unfinished_sents + (self.pad_idx) * (1 - unfinished_sents)
                
            #"""
            """
            if not self.no_template:
                tokens_to_add = next_token.squeeze() * no_template_pos + cur_template *  ~(no_template_pos)
                tokens_to_add = tokens_to_add * unfinished_sents + (self.pad_idx) * ~(unfinished_sents)
            else:
                tokens_to_add = next_token.squeeze() * unfinished_sents + (self.pad_idx) * ~(unfinished_sents)
            """
            #print('tokens_to_add',tokens_to_add)
            
            if not self.quiet:
                output_str = ''
                for b in range(batch_size):
                    w = self.tokenizer.convert_ids_to_tokens([tokens_to_add[b]])[0]
                    #print(w)
                    p = probs[b][-1]
                    output_str += '{:>12}({:.2f})|'.format(w, 100 * p)
                if cur_len == 1:
                    print('=' * 50)
                print('step={:<3d}|{}'.format(cur_len, output_str))

            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            if cur_len < max_template_len:
                template[:, cur_len-1] = tokens_to_add
            last = input_ids[:, -1:]
            eos_in_sents = tokens_to_add == self.eos_idx
            unfinished_sents.mul_((~eos_in_sents).long())

            
            
            edu_in_sents = tokens_to_add == self.edu_idx
            for b in range(batch_size):
                _edu_in_sents = edu_in_sents[b].item() 
                if _edu_in_sents == 1:
                    cur_edu_index = cur_edu_indices[b]
                    all_edu_hidden_states[b, cur_edu_index] = last_hidden_state[b, -1, :]
            
            if unfinished_sents.max() == 0:
                break
            
            
            cur_len = cur_len + 1

        return input_ids, probs

