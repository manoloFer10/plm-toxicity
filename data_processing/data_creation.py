import pandas as pd
from datasets import load_dataset
from utils.visualize_activations import filter_by_top_taxa, get_rank
from data_processing.cd_hit import map_sequences_to_cdhit_clusters, pick_cluster_representatives_from_df

UNIPROT_CURATED = load_dataset('manufernandezbur/uniprot2025_03')['train']

def add_eos(example):
        example['Sequence'] =  '<|endoftext|>' + example['Sequence'] + '<|endoftext|>'
        return example

def get_splits(data):
        
    tox = data.filter(lambda example: example['Toxin'] and 10<example['Length']<1000)
    non_tox = data.filter(lambda example: not example['Toxin'] and 10<example['Length']<1000)
    
    return tox.to_pandas(), non_tox.to_pandas()

def get_representatives_within_taxa(ranked_df, rank, identity):
    taxas = list(ranked_df[rank].unique())

    dfs= []
    for taxa in taxas:
        filtered = ranked_df[ranked_df[rank]==taxa]
        df = map_sequences_to_cdhit_clusters(filtered, identity=identity)
        df = pick_cluster_representatives_from_df(df)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main():
    rank = 'species'
    identity = 0.9

    tox, non_tox = get_splits(UNIPROT_CURATED)

    #Filter by taxa present in toxic proteins
    tox_rank, non_tox_rank = filter_by_top_taxa(tox, non_tox,
                       rank=rank,
                       n_top='all', #every species in tox.
                       lineage_col="Taxonomic lineage",
                       match_counts=False, #keep all proteins! clustering will trim most.
                       random_state=42)
    print(f'Filtered by top taxa in toxic set: {len(tox_rank[rank].unique())} unique taxa.')
    
    #Cluster
    tox_clustered_by_taxa = get_representatives_within_taxa(tox_rank, rank, identity=identity)
    print('Finished tox.')
    non_tox_clustered_by_taxa = get_representatives_within_taxa(non_tox_rank, rank, identity=identity)
    print('Finished non-tox.')

    #Save
    tox_clustered_by_taxa.to_csv('clustered_toxins.csv', index=False)
    non_tox_clustered_by_taxa.to_csv('clustered_non_toxins.csv', index = False)
    
if __name__ == '__main__':
    main()