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
from src.utils.generate_utils import mask_for_de_novo
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.lm.memdlm.diffusion_module import MembraneDiffusion
from src.sampling.multipass_sampler import MultipassSampler

from src.utils.generate_utils import calc_ppl
from src.utils.model_utils import _print
from src.utils.config_utils import load_config, repo_path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = load_config("multipass.yaml")

date = datetime.now().strftime("%Y-%m-%d")




def main():
    csv_save_path = repo_path(
        'results', 'multipass', config.wandb.name, date,
        f"lamb={config.guidance.reg_strength}_tau={config.guidance.sampling_temperature}"
    )
    
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

    generator = MultipassSampler(config, device, memdlm, tokenizer)

    seq_lengths = [x for x in range(60, 161) for _ in range(3)]
    #seq_lengths = random.sample([x for x in range(120, 161)], 15)

    generation_results = []
    for seq_len in tqdm(seq_lengths, desc=f"Generating sequences: "):
        seq_res = []

        masked_seq = mask_for_de_novo(seq_len) # Sequence of all <mask> tokens
        tokens = tokenizer(masked_seq, return_tensors='pt')

        gen_seq = ""
        attempts = 0

        while len(gen_seq) != seq_len and attempts < 3:
            gen_seq, pred_tm_segs = generator.sample_guidance(
                tokens,
                config.guidance.guide_steps,
                config.guidance.diffusion_steps
            )
            attempts += 1

        if len(gen_seq) != seq_len:
            esm_ppl, memdlm_ppl = None, None
        else:
            esm_ppl = calc_ppl(esm_model, tokenizer, gen_seq, [i for i in range(len(gen_seq))], model_type='esm')
            memdlm_ppl = calc_ppl(memdlm, tokenizer, gen_seq, [i for i in range(len(gen_seq))], model_type='diffusion')

        _print(f'seq: {gen_seq}')
        _print(f'pred_tm_segs: {pred_tm_segs}')
        _print(f"ESM PPL: {esm_ppl}")
        _print(f"MeMDLM PPL: {memdlm_ppl}")
        _print('\n')

        seq_res.append(gen_seq)
        seq_res.append(esm_ppl)
        seq_res.append(memdlm_ppl)
        seq_res.append(pred_tm_segs)
        generation_results.append(seq_res)

    df = pd.DataFrame(generation_results, columns=['Generated Sequence', 'ESM PPL', 'MeMDLM PPL', 'Pred TM Segments'])
    df.to_csv(str(csv_save_path / "seqs_with_ppl.csv"), index=False)


if __name__ == "__main__":
    main()