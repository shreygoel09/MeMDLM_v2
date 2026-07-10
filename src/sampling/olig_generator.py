#!/usr/bin/env python3

import sys
import os

import random
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.lm.memdlm.diffusion_module import MembraneDiffusion
from src.sampling.olig_sampler import NOSSampler

from src.utils.generate_utils import calc_blosum_score, calc_ppl
from src.utils.model_utils import _print
from src.utils.config_utils import load_config, repo_path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = load_config("oligo.yaml")

date = datetime.now().strftime("%Y-%m-%d")




def main():
    csv_save_path = repo_path('results', 'oligo', config.wandb.name, date)
    
    try: os.makedirs(csv_save_path, exist_ok=False)
    except FileExistsError: pass

    tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_evoflow)
    
    memdlm = MembraneDiffusion(config).to(device)
    state_dict = memdlm.get_state_dict(str(repo_path("checkpoints", config.lm.ft_evoflow, "best_model.ckpt")))
    memdlm.load_state_dict(state_dict)
    memdlm.eval()

    esm_pth = config.lm.pretrained_esm
    esm_model = AutoModelForMaskedLM.from_pretrained(esm_pth).to(device)
    esm_model.eval()

    generator = NOSSampler(config, device, memdlm, esm_model, tokenizer)

    # Determine length from positive controls
    df = pd.read_csv(str(repo_path('data', 'olig_clf', 'test.csv')))
    seqs = df['Sequence'].tolist()


    generation_results = []
    for seq in tqdm(seqs, desc=f"Generating sequences: "):
        seq_res = []

        seq_len = len(seq)
        tokens = tokenizer(seq, return_tensors='pt')

        gen_seq = ""
        attempts = 0

        while len(gen_seq) != seq_len and attempts < 3:
            gen_seq, og_pred, final_pred = generator.sample_guidance(
                tokens,
                config.olig_guidance.guide_steps,
                config.olig_guidance.diffusion_steps
            )
            attempts += 1

        if len(gen_seq) != seq_len:
            esm_ppl, memdlm_ppl = None, None
        else:
            esm_ppl = calc_ppl(esm_model, tokenizer, gen_seq, [i for i in range(len(gen_seq))], model_type='esm')
            memdlm_ppl = calc_ppl(memdlm, tokenizer, gen_seq, [i for i in range(len(gen_seq))], model_type='diffusion')
            blosum = calc_blosum_score(seq, gen_seq, indices=[i for i in range(len(gen_seq))])

        seq_res.append(seq)
        seq_res.append(gen_seq)
        seq_res.append(og_pred)
        seq_res.append(final_pred)
        seq_res.append(final_pred - og_pred)
        seq_res.append(esm_ppl)
        seq_res.append(memdlm_ppl)
        seq_res.append(blosum)
        generation_results.append(seq_res)

    df = pd.DataFrame(generation_results, columns=['Original Sequence', 'Generated Sequence', 'OG Olig Value', 'New Olig Value', 'Olig Increase', 'ESM PPL', 'MeMDLM PPL', 'MemDLM Blosum'])
    df.to_csv(str(csv_save_path / "seqs_with_ppl.csv"), index=False)


if __name__ == "__main__":
    main()