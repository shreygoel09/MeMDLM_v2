#!/usr/bin/env python3

import sys
import os

import random
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import Counter
from omegaconf import OmegaConf
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM

from MeMDLM_v2.src.lm.diffusion_module import MembraneFlow
from src.sampling.unconditional_sampler import UnconditionalSampler
from src.utils.generate_utils import mask_for_de_novo, calc_ppl
from src.utils.model_utils import _print


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir('/home/a03-sgoel/MeMDLM_v2')
config = OmegaConf.load("./src/configs/lm.yaml")

date = datetime.now().strftime("%Y-%m-%d")



def generate_sequence(prior: str, tokenizer, generator, device):
    input_ids = tokenizer(prior, return_tensors="pt").to(device)['input_ids']
    ids = generator.sample_unconditional(
        xt=input_ids,
        num_steps=config.sampling.n_steps,
        return_logits=False,
        banned_token_ids=None
        #banned_token_ids=[tokenizer.convert_tokens_to_ids("P"), tokenizer.convert_tokens_to_ids("C")]
    )
    generated_sequence = tokenizer.decode(ids[0].squeeze())[5:-5].replace(" ", "") # bos/eos tokens & spaces between residues
    return generated_sequence


def main():
    csv_save_path = f'./results/denovo/unconditional/{config.wandb.name}/{date}_tau=3.0_test-set_distribution'
    
    try: os.makedirs(csv_save_path, exist_ok=False)
    except FileExistsError: pass


    tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_evoflow)
    
    flow = MembraneFlow(config).to(device)
    state_dict = flow.get_state_dict(f"./checkpoints/{config.wandb.name}/best_model.ckpt")
    flow.load_state_dict(state_dict)
    flow.eval()

    esm_pth = config.lm.pretrained_esm
    esm_model = AutoModelForMaskedLM.from_pretrained(esm_pth).to(device)
    esm_model.eval()

    generator = UnconditionalSampler(tokenizer, flow)

    # # Get 100 random sequence lengths to generate
    # seq_lengths = [random.randint(50, 250) for _ in range(5000)]

    # # Determine length from positive controls
    # df = pd.read_csv(f'./results/denovo/unconditional/{config.wandb.name}/perin_pos_ctrl/raw_seqs.csv')
    # seq_lengths = [len(seq) for seq in df['Sequence'].tolist() for _ in range(500)] # generate each length 100 times
    # _print(seq_lengths)

    # Determine lengths from test set distribution
    df = pd.read_csv("./data/test.csv")
    seq_lengths = [len(seq) for seq in df['Sequence'].tolist()]
    length_counts = Counter(seq_lengths) # {L1: freq, L2: freq, ...}
    total = sum(length_counts.values()) # total number of tokens
    lengths = np.array(list(length_counts.keys())) # Frequency of each length
    probs = np.array([length_counts[l] / total for l in lengths])
    seq_lengths = np.random.choice(lengths, size=len(seq_lengths), p=probs)

    generation_results = []
    for seq_len in tqdm(seq_lengths, desc=f"Generating sequences: "):
        seq_res = []

        masked_seq = mask_for_de_novo(seq_len) # Sequence of all <mask> tokens
        gen_seq = ""
        attempts = 0

        while len(gen_seq) != seq_len and attempts < 3:
            gen_seq = generate_sequence(masked_seq, tokenizer, generator, device)
            attempts += 1

        if len(gen_seq) != seq_len:
            esm_ppl, flow_ppl = None, None
        else:
            esm_ppl = calc_ppl(esm_model, tokenizer, gen_seq, [i for i in range(len(gen_seq))], model_type='esm')
            flow_ppl = calc_ppl(flow, tokenizer, gen_seq, [i for i in range(len(gen_seq))], model_type='flow')
        
        _print(f'gen seq: {gen_seq}')
        _print(f'esm ppl: {esm_ppl}')
        _print(f'flow ppl: {flow_ppl}')
        
        seq_res.append(gen_seq)
        seq_res.append(esm_ppl)
        seq_res.append(flow_ppl)

        generation_results.append(seq_res)

    df = pd.DataFrame(generation_results, columns=['Generated Sequence', 'ESM PPL', 'Flow PPL'])
    df.to_csv(csv_save_path + "/seqs_with_ppl.csv", index=False)


if __name__ == "__main__":
    main()