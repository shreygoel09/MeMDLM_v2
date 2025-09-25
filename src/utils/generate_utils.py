import torch
import math
import sys

import torch.nn.functional as F
import pandas as pd
import numpy as np

from omegaconf import OmegaConf
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer

from src.lm.memdlm.diffusion_module import MembraneFlow
from src.lm.dplm.diffusion_module import DPLM
from src.utils.model_utils import get_latents, _print
from src.sampling.unconditional_sampler import UnconditionalSampler
from src.lm.dplm.unconditional_sampler import UnconditionalSampler as DPLMUnconditionalSampler

config = OmegaConf.load("/home/a03-sgoel/MeMDLM_v2/src/configs/lm.yaml")

# -------# Masking #-------- #
def mask_for_de_novo(sequence_length):
    return "<mask>" * sequence_length

def mask_for_scaffold(sequence, generate_type, mask_token):
    if generate_type == "uppercase":
        sequence = ''.join([mask_token if residue.isupper() else residue.upper() for residue in sequence])
    elif generate_type == "lowercase":
        sequence = ''.join([mask_token if residue.islower() else residue for residue in sequence])   
    return sequence


# -------# Generation #-------- #
def memflow_infill_uncond(masked_seq, tokenizer, model: MembraneFlow):
    generator = UnconditionalSampler(tokenizer, model) # initialize the generator object
    xt = tokenizer(masked_seq, return_tensors='pt')['input_ids'].to(model.device)
    denoised_tokens = generator.sample_unconditional(xt, config.sampling.n_steps)[0].squeeze()
    generated_sequence = tokenizer.decode(denoised_tokens).replace(" ", "")[5:-5]
    return generated_sequence


def evodiff_infill(motif_seq, tokenizer, model, device, batch_size=1):
    """
    Following the given evodiff example
    https://github.com/microsoft/evodiff/blob/main/examples/evodiff.ipynb
    """    
    # Manual masking of infilling sequence
    motif_seq = ''.join(["#" if aa.islower() else aa for aa in motif_seq])  # Mask token is "#" in evodiff tokenizer
    tkns = tokenizer.tokenize([motif_seq])
    sample = torch.as_tensor(tkns).to(device)

    # Create input motif + scaffold
    loc = torch.arange(0, len(motif_seq)).to(device)[sample==tokenizer.mask_id].cpu().numpy()
    np.random.shuffle(loc)
    
    sample = sample.to(device).unsqueeze(0)
    # og_sample = sample.clone()
    
    with torch.no_grad():
        for i in loc:
            timestep = torch.tensor([0] * batch_size).to(device)  # placeholder but not called in model
            timestep = timestep.to(device)
            prediction = model(sample, timestep)
            p = prediction[:, i, :len(tokenizer.all_aas) - 6]  # only canonical
            p = F.softmax(p, dim=1)  # softmax over logits
            p_sample = torch.multinomial(p, num_samples=1) # sample from categorical distribution
            sample[:, i] = p_sample.squeeze()
    output = [tokenizer.untokenize(s) for s in sample]
    return output[0] #if batch_size==1 else output, og_sample, loc


def dplm_infill(masked_seq, tokenizer, model: DPLM, device):
    generator = DPLMUnconditionalSampler(tokenizer, model)
    xt = tokenizer(masked_seq, return_tensors='pt')['input_ids'].to(model.device)
    denoised_tokens = generator.sample_unconditional(xt, config.sampling.n_steps)[0].squeeze()
    generated_sequence = tokenizer.decode(denoised_tokens).replace(" ", "")[5:-5]
    return generated_sequence


# -------# Metrics #-------- #
def calc_progen_ppl(model, tokenizer, target, device, fp16=True):
    """Compute causal LM cross-entropy loss for a given sequence."""
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(
                input_ids = target,
                attention_mask = torch.ones_like(target)
            ).logits
            # Shift
            logits = logits[:-1, ...]
            target = target[1:]
            loss = torch.nn.functional.cross_entropy(
                input=logits,
                target=target,
                reduction='mean'
            )
            return torch.exp(loss).item()


def calc_ppl(model, tokenizer, generated_sequence, mask_token_indices, model_type):
    total_loss = 0.0
    tensor_input = tokenizer.encode(generated_sequence, return_tensors='pt').to(model.device)
    attn_mask = torch.ones_like(tensor_input).to(model.device)

    for i in mask_token_indices:
        masked_input = tensor_input.clone()
        masked_input[0, i] = tokenizer.mask_token_id
    
        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, i] = tensor_input[0, i]

        with torch.no_grad():
            if model_type == 'esm':
                loss = model(masked_input, labels=labels).loss.item()
            elif model_type == 'flow':
                logits = model.forward(masked_input, attention_mask=attn_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='none',
                    ignore_index=-100,
                )[i].item()
 
            total_loss += loss
    
    avg_loss = total_loss / len(generated_sequence)
    perplexity = math.exp(avg_loss)

    return perplexity


def calc_blosum_score(og_seq, gen_seq, indices):
    import blosum as bl
    mat = bl.BLOSUM(62)
    tot_score = 0
    for i in indices:
        og_res, gen_res = og_seq[i], gen_seq[i]
        try:
            val = mat[og_res][gen_res]
            tot_score += val
        except KeyError:
             # -4 is lowest BLOSUM score indicating biological implausability
            tot_score += -4
    return tot_score / len(indices) if indices else 0


def calc_cos_sim(original_sequence, generated_sequence, tokenizer, esm_model, device):
    og_embeddings = get_latents(esm_model, tokenizer, original_sequence.upper(), device)
    new_embeddings = get_latents(esm_model, tokenizer, generated_sequence, device)
    cosine_sim = torch.nn.functional.cosine_similarity(og_embeddings, new_embeddings, dim=-1)
    cosine_sim = torch.mean(cosine_sim).item()
    return cosine_sim