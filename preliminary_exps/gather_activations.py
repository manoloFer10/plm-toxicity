from preliminary_exps.utils.extract_activations import get_activations_for_datasets
from utils.models import get_protgpt2
from datasets import load_dataset
import torch


def get_data_from_activations(acts_tox, acts_non_tox, device = 'cpu'):
    acts_combined = torch.cat([acts_tox, acts_non_tox], dim=0)   # [Nt+Nn, P, L, d]

    labels = torch.cat([
        torch.ones(acts_tox.shape[0], dtype=int, device = device),     # 1 = tox
        torch.zeros(acts_non_tox.shape[0], dtype=int, device = device)      # 0 = non-tox
    ])

    return acts_combined, labels


def get_datasets():
    def add_eos(example):
        example['Sequence'] = example['Sequence'] + '<|endoftext|>'
        return example
    
    data=load_dataset('manufernandezbur/uniref50')['train']
    
    tox = data.filter(lambda example: example['Toxin'])
    non_tox = data.filter(lambda example: not example['Toxin'])

    tox = tox.map(add_eos)
    non_tox = non_tox.map(add_eos)
    
    return tox, non_tox


if __name__ == '__main__':
    model, tokenizer = get_protgpt2()

    tox, non_tox = get_datasets()

    tox_seqs = list(tox['Sequence'])
    non_tox_seqs = list(non_tox['Sequence'])

    acts_tox, acts_non_tox = get_activations_for_datasets(model, 
                                                          tokenizer, 
                                                          tox_seqs, 
                                                          non_tox_seqs, 
                                                          block_modules = model.transformer.h)




