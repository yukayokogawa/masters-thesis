import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from .strategy_utils import (
    bart_past_for_pplm,
    pplm_past_for_bart,
    get_nucleus_satellite_hidden_states,
)



PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def perturb_past(
        past_key_values=None,
        model=None,
        last=None,
        unpert_past_key_values=None,
        unpert_logits=None,
        encoder_hidden_states=None,
        attention_mask=None,
        relation_labels=None,
        nuclearity=None,
        edu_hidden_states=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR,
):
    #print(edu_hidden_states.shape)
    
    past, enc_past = bart_past_for_pplm(past_key_values)


    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        perturbed_past_key_values = pplm_past_for_bart(perturbed_past, enc_past)
        
        perturbed_outputs = model(
            decoder_input_ids=last,
            past_key_values=perturbed_past_key_values,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
        )
        outputs = perturbed_outputs['outputs']
        all_logits = outputs['logits']
        all_hidden = outputs['decoder_hidden_states']
        hidden_states = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden_states,
            dim=1
        ).detach()
        generated_hidden = hidden_states[:, -1:].clone().detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(logits, dim=-1)
        #print(next_token)
        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:

            ce_loss = torch.nn.CrossEntropyLoss(
                ignore_index=0,
            )
            nucleus_hidden_states, satellite_hidden_states = get_nucleus_satellite_hidden_states(
                generated_hidden,
                edu_hidden_states,
                nuclearity,
            )
            prediction = classifier(nucleus_hidden_states, satellite_hidden_states)
            prediction = prediction.view(-1, prediction.size(-1))
            label = relation_labels.view(-1)
            discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits, dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward(retain_graph=True)

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

        past_key_values = pplm_past_for_bart(past, enc_past)






    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))
    pert_past_key_values = pplm_past_for_bart(pert_past, enc_past)

    return pert_past_key_values, new_accumulated_hidden, grad_norms, loss_per_iter

