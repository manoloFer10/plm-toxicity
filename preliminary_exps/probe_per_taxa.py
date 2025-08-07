from preliminary_exps.utils.extract_activations import get_activations_for_datasets
from preliminary_exps.utils.probing import layerwise_linear_probe, layerwise_rsa, fisher_ratio
from utils.models import get_protgpt2
from preliminary_exps.utils.visualize_activations import filter_by_top_taxa, save_class_signal_plot
from datasets import load_dataset
import torch
import os
import pandas as pd


def get_data_from_activations(acts_tox, acts_non_tox, device = 'cpu'):
    acts_combined = torch.cat([acts_tox, acts_non_tox], dim=0)   # [Nt+Nn, P, L, d]

    labels = torch.cat([
        torch.ones(acts_tox.shape[0], dtype=int, device = device),     # 1 = tox
        torch.zeros(acts_non_tox.shape[0], dtype=int, device = device)      # 0 = non-tox
    ])

    return acts_combined.cpu(), labels #return everything on cpu for probing algorithms


def benchmark_data(dfs, model, tokenizer, taxa):
    tox = dfs[0]
    non_toxs = dfs[1]
    tox_sequences = list(tox['Sequence'])
    non_tox_sequences = list(non_toxs['Sequence'])

    #Extract activations
    acts_tox, acts_non_tox = get_activations_for_datasets(model, 
                                                          tokenizer, 
                                                          tox_sequences, 
                                                          non_tox_sequences, 
                                                          block_modules = model.transformer.h,
                                                          batch_size = 96)
    
    #Transform data for probing
    acts_combined, labels = get_data_from_activations(
        acts_tox=acts_tox,
        acts_non_tox=acts_non_tox)
    
    #Test
    metrics, clfs = layerwise_linear_probe(acts_combined, labels)
    f_ratios = fisher_ratio(acts_combined, labels)
    rsa_scores = layerwise_rsa(acts_combined, labels)

    # Save plot
    out_path= f"results/probing/within_taxa/{taxa}_signal_layers.png"
    os.makedirs(out_path, exist_ok=True)

    path = save_class_signal_plot(
            acc = metrics['accuracy'],
            auc = metrics['auc'],
            f1 = metrics['f1'],
            fisher= f_ratios,
            rsa_scores = rsa_scores,
            out_path= out_path)
    print(f'Saved figure results of {taxa} to {path}.')

    #return clfs # for pairwise evaluations.
    

#def benchmark_all(regressor, taxa, dfs):



def get_datasets():
    def add_eos(example):
        example['Sequence'] = example['Sequence'] + '<|endoftext|>'
        return example
    
    data=load_dataset('manufernandezbur/uniref50')['train']
    
    tox = data.filter(lambda example: example['Toxin'] and example['Length']<300)
    non_tox = data.filter(lambda example: not example['Toxin'] and example['Length']<300)

    tox = tox.map(add_eos)
    non_tox = non_tox.map(add_eos)
    
    return tox.to_pandas(), non_tox.to_pandas()


def main():
    model, tokenizer = get_protgpt2()

    tox, non_tox = get_datasets()

    # tox_seqs = list(tox['Sequence'])
    # non_tox_seqs = list(non_tox['Sequence'])

    # acts_tox, acts_non_tox = get_activations_for_datasets(model, 
    #                                                       tokenizer, 
    #                                                       tox_seqs, 
    #                                                       non_tox_seqs, 
    #                                                       block_modules = model.transformer.h,
    #                                                       batch_size = 96)
    
    rank = "phylum"

    tox_fam, non_tox_fam = filter_by_top_taxa(
        tox, non_tox,
        rank=rank,
        n_top=9,
        lineage_col="Taxonomic lineage",
        match_counts=True     # False if you just want to keep *all* matches
    )

    top_taxa = tox_fam[rank].unique()

    filtered_by_taxa = {
        taxa: (tox_fam[tox_fam[rank] == taxa], non_tox_fam[non_tox_fam[rank] == taxa]) 
        for taxa in top_taxa
    }

    # Benchmark logreg on taxa and retrieve 
    regressors = {}
    for taxa, dfs in filtered_by_taxa.values():
        benchmark_data(dfs, model, tokenizer, taxa)
        #clfs = benchmark_data(dfs, model, tokenizer, taxa)
        #regressors[taxa] = clfs

    # # Benchmark logreg 1-1
    # for taxa, regressor in regressors.values():
    #     benchmark_all(regressor, taxa, filtered_by_taxa)


if __name__ == '__main__':
    main()

