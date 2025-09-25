#!/usr/bin/env python3

import sys
import os
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.lm.memdlm.diffusion_module import MembraneFlow
from src.utils.model_utils import _print
from src.sampling.guided_sampler import GuidedSampler
from src.utils.generate_utils import (
    mask_for_scaffold,
    calc_blosum_score,
    calc_ppl
)

config = OmegaConf.load("/home/a03-sgoel/MeMDLM_v2/src/configs/guidance.yaml")

os.chdir(f'/home/a03-sgoel/MeMDLM_v2/results/infilling/guided/{config.lm.ft_evoflow}/test_set/')
todays_date = datetime.today().strftime('%Y-%m-%d')
csv_save_path = f'./{todays_date}_boltzmann-soft_new_clf_data_cleaned/'
try: os.makedirs(csv_save_path, exist_ok=False)
except FileExistsError: pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_esm)
    esm_model = AutoModelForMaskedLM.from_pretrained(config.lm.pretrained_esm).eval().to(device)

    diffusion = MembraneFlow(config).to(device)
    state_dict = diffusion.get_state_dict(f"/home/a03-sgoel/MeMDLM_v2/checkpoints/{config.lm.ft_evoflow}/best_model.ckpt")
    diffusion.load_state_dict(state_dict)
    diffusion.eval().to(device)

    sampler = GuidedSampler(config, esm_model, tokenizer, diffusion, device)

    df = pd.read_csv('/home/a03-sgoel/MeMDLM_v2/data/classifier/test.csv')
    sequences = df['Sequence'].tolist()

    gen_seqs, ppls, blosums = [], [], []


    for seq in tqdm(sequences, desc='Infilling Sequences'):
        masked_seq = mask_for_scaffold(seq, generate_type='uppercase', mask_token='<mask>')
        tokens = tokenizer(masked_seq, return_tensors='pt')
        input_ids, attn_masks = tokens['input_ids'].to(device), tokens['attention_mask'].to(device)
        
        soluble_idxs = [i for i in range(len(seq)) if seq[i].isupper()]
        infilled_tokens = sampler.optimize_sequence(
            input_ids=input_ids,
            attn_masks=attn_masks,
            soluble_indices=soluble_idxs,
        )
        infilled_seq = tokenizer.decode(infilled_tokens).replace(" ", "")[5:-5]
        
        bl = calc_blosum_score(seq.upper(), infilled_seq, soluble_idxs)
        try:
            ppl = calc_ppl(esm_model, tokenizer, infilled_seq, [i for i in range(len(seq))], model_type='esm')
        except:
            ppl = float('inf')

        gen_seqs.append(infilled_seq)
        ppls.append(ppl)
        blosums.append(bl)

        _print(seq)
        _print(infilled_seq)
        _print(ppl)
        _print(bl)
        _print('\n')


    df['MeMDLM Sequence'] = gen_seqs
    df['MeMDLM PPL'] = ppls
    df['MeMDLM BLOSUM'] = blosums

    _print(df)
    df.to_csv(f'./{csv_save_path}/t=0.7_new-data-cleaned_infilled_seqs.csv', index=False)


    
if __name__ == "__main__":
    main()
    
