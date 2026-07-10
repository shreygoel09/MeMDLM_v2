#!/usr/bin/env python3


import os
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.lm.memdlm.diffusion_module import MembraneDiffusion
from src.sampling.pet_sampler import PETSampler
from src.utils.generate_utils import (
    mask_for_scaffold,
    calc_blosum_score,
    calc_ppl,
    calc_tm_enrich
)

from src.utils.model_utils import _print
from src.utils.config_utils import load_config, repo_path


config = load_config("desolubilize.yaml")
task = config.guidance.task

results_root = repo_path("results", "heme", config.lm.ft_evoflow)
todays_date = datetime.today().strftime('%Y-%m-%d')

if config.guidance.prior == 'boltzmann':
    csv_save_path = results_root / task / todays_date / f"{config.guidance.prior}-t={config.guidance.boltzmann_t}_p={config.guidance.top_p}_tau={config.guidance.sampling_t}"
elif config.guidance.prior == 'lm_probs':
    csv_save_path = results_root / task / todays_date / f"{config.guidance.prior}_p={config.guidance.top_p}_tau={config.guidance.sampling_t}"

try: os.makedirs(csv_save_path, exist_ok=False)
except FileExistsError: pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_esm)
    esm_model = AutoModelForMaskedLM.from_pretrained(config.lm.pretrained_esm).eval().to(device)

    diffusion = MembraneDiffusion(config).to(device)
    state_dict = diffusion.get_state_dict(str(repo_path("checkpoints", config.lm.ft_evoflow, "best_model.ckpt")))
    diffusion.load_state_dict(state_dict)
    diffusion.eval().to(device)

    sampler = PETSampler(config, esm_model, tokenizer, diffusion, device)

    # Update this path to your input CSV of scaffold sequences (uppercase = TM, lowercase = soluble).
    df = pd.read_csv(str(repo_path("results", "heme", "4d2.csv")))
    sequences = df['Sequence'].tolist()

    gen_seqs, ppls, blosums, og_tms, gen_tms, delta_tms = [], [], [], [], [], []


    for seq in tqdm(sequences, desc='Desolubilizing Sequences'):
        masked_seq = mask_for_scaffold(seq, generate_type='uppercase', mask_token='<mask>')
        tokens = tokenizer(masked_seq, return_tensors='pt')
        input_ids, attn_masks = tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

        tm_idxs = [i for i in range(len(seq)) if seq[i].isupper()]
        soluble_idxs = [i + 1 for i in range(len(seq)) if seq[i].islower()]

        infilled_tokens = sampler.optimize_sequence(
            input_ids=input_ids,
            attn_masks=attn_masks,
            soluble_indices=soluble_idxs,
        )
        infilled_seq = tokenizer.decode(infilled_tokens).replace(" ", "")[5:-5]
        
        try:
            bl = calc_blosum_score(seq.upper(), infilled_seq, tm_idxs)
        except:
            bl = float('inf')
        
        try:
            ppl = calc_ppl(esm_model, tokenizer, infilled_seq, [i for i in range(len(seq))], model_type='esm')
        except:
            ppl = float('inf')

        try:
            og_tm, gen_tm, delta_tm = calc_tm_enrich(seq.upper(), infilled_seq, tm_idxs)
        except:
            og_tm, gen_tm, delta_tm = float('inf'), float('inf'), float('inf')

        gen_seqs.append(infilled_seq)
        ppls.append(ppl)
        blosums.append(bl)
        og_tms.append(og_tm)
        gen_tms.append(gen_tm)
        delta_tms.append(delta_tm)

        _print(seq)
        _print(infilled_seq)
        _print(ppl)
        _print(bl)
        _print(og_tm)   
        _print(gen_tm)
        _print(delta_tm)
        _print('\n')


    df['MeMDLM Sequence'] = gen_seqs
    df['MeMDLM PPL'] = ppls
    df['MeMDLM BLOSUM'] = blosums
    df['OG TM Enrichment'] = og_tms
    df['MeMDLM TM Enrichment'] = gen_tms
    df['Delta TM Enrichment'] = delta_tms

    _print(df)
    df.to_csv(csv_save_path / "infilled_seqs.csv", index=False)


    
if __name__ == "__main__":
    main()
