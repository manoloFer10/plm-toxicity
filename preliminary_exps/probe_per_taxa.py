from preliminary_exps.utils.extract_activations import get_activations_for_datasets
from preliminary_exps.utils.probing import layerwise_linear_probe, layerwise_rsa, fisher_ratio
from utils.models import get_protgpt2
from preliminary_exps.utils.visualize_activations import filter_by_top_taxa, save_class_signal_plot
from datasets import load_dataset
from tqdm import tqdm
import torch
import os
import pandas as pd


def get_data_from_activations(acts_tox, acts_non_tox, device = 'cpu'):
    acts_combined = torch.cat([acts_tox, acts_non_tox], dim=0).cpu()   # [Nt+Nn, P, L, d]

    labels = torch.cat([
        torch.ones(acts_tox.shape[0], dtype=int, device = device),     # 1 = tox
        torch.zeros(acts_non_tox.shape[0], dtype=int, device = device)      # 0 = non-tox
    ])

    return acts_combined, labels #return everything on cpu for probing algorithms


def get_acts(dfs, model, tokenizer):
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

    return acts_tox, acts_non_tox


def benchmark_data(acts_tox, acts_non_tox, taxa, taxonomy, is_full=False):
    
    #Define prefix
    if is_full:
        prefix='Full_'
    else:
        prefix=''

    #Transform data for probing
    acts_combined, labels = get_data_from_activations(
        acts_tox=acts_tox,
        acts_non_tox=acts_non_tox)
    
    #Test
    metrics = layerwise_linear_probe(acts_combined, labels)
    #f_ratios = fisher_ratio(acts_combined, labels)
    rsa_scores = layerwise_rsa(acts_combined, labels)

    # Save plot
    out_path= f"results/probing/within_taxa/{taxonomy}/{prefix}{taxa}_signal_layers.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    path_fig = save_class_signal_plot(
            acc = metrics['accuracy'],
            auc = metrics['auc'],
            f1 = metrics['f1'],
            #fisher= f_ratios,
            rsa_scores = rsa_scores,
            out_path= out_path)
    
    #Save CSV
    out_path = f'results/probing/within_taxa/{taxonomy}/{prefix}{taxa}_probing_results_per_layer.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results = pd.DataFrame(metrics)
    df_results.T.rename_axis("metric").to_csv(out_path, index=True) # rows as metrics


    print(f'Saved results of {taxa} to {path_fig} and {out_path}.')

    #return acts_combined, labels
    

#def benchmark_all(regressor, taxa, dfs):



def get_datasets():
    def add_eos(example):
        example['Sequence'] =  '<|endoftext|>' + example['Sequence'] + '<|endoftext|>'
        return example
    
    data=load_dataset('manufernandezbur/uniref50')['train']
    
    tox = data.filter(lambda example: example['Toxin'] and example['Length']<1000)
    non_tox = data.filter(lambda example: not example['Toxin'] and example['Length']<1000)

    tox = tox.map(add_eos)
    non_tox = non_tox.map(add_eos)
    
    return tox.to_pandas(), non_tox.to_pandas()


def main():
    model, tokenizer = get_protgpt2()

    tox, non_tox = get_datasets()
    
    rank = "species"

    tox_fam, non_tox_fam = filter_by_top_taxa(
        tox, non_tox,
        rank=rank,
        n_top=15,
        lineage_col="Taxonomic lineage",
        match_counts=True     # False if you just want to keep *all* matches
    )

    top_taxa = tox_fam[rank].unique()

    filtered_by_taxa = {
        taxa: (tox_fam[tox_fam[rank] == taxa], non_tox_fam[non_tox_fam[rank] == taxa]) 
        for taxa in top_taxa
    }

    # Benchmark logreg on taxa and retrieve 
    unprocessed={}
    all_tox_acts, all_nontox_acts = [], []
    for taxa, dfs in tqdm(filtered_by_taxa.items(), total = len(filtered_by_taxa), desc = 'Probing taxa'):
        if len(dfs[0])<2 or len(dfs[1])<2 :
            unprocessed[taxa]= dfs
            continue
        else:
            #Extract activations
            acts_tox, acts_non_tox = get_acts(dfs, model, tokenizer)
            all_tox_acts.append(acts_tox)
            all_nontox_acts.append(acts_non_tox)

            benchmark_data(acts_tox, acts_non_tox, taxa, rank)


    def _print_unprocessed(unprocessed):
        for k,v in unprocessed.items():
            print(f'Taxa: {k}, Tox:{len(v[0])}, NonTox:{len(v[1])}')

    print(f'Some dataframes had no samples: \n {_print_unprocessed(unprocessed)}.')
    unprocessed_non_tox = sum([len(v[1]) for k,v in unprocessed.items()])
    unprocessed_tox = sum([len(v[0]) for k,v in unprocessed.items()])

    benchmark_data(all_tox_acts, all_nontox_acts,taxa, rank, is_full=True)

    print(f'Processed {len(tox_fam)-unprocessed_tox} toxic samples and {len(non_tox_fam)-unprocessed_non_tox} non-toxic samples.') 
    # Benchmark logreg 1-1
    # for taxa, regressor in regressors.items():
    #     benchmark_all(regressor, taxa, filtered_by_taxa)


if __name__ == '__main__':
    main()

