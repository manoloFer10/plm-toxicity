
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # stable indexing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # only show GPU 0 to this process

import pandas as pd
import argparse
from utils.scoring import score_toxicity, calculatePerplexity
from utils.models.protgpt2 import clean_protgpt2_generation


def get_most_viable(model, sequences, top_k=100, batch_size = 8):
    
    tokenizer_fn = model.tokenize_instructions_fn
    
    top_k = min(top_k, len(sequences))

    scored = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        ppls = calculatePerplexity(batch_sequences, model, tokenizer_fn)
        ppls = ppls.tolist() if hasattr(ppls, "tolist") else list(ppls)
        scored.extend(zip(batch_sequences, ppls))
    

    scored.sort(key=lambda x: x[1])  # sort by perplexity
    scored = scored[:top_k]  # keep only top_k sequences with lowest perplexity

    sequences = [s for s, _ in scored]
    ppls = [p for _, p in scored]
    return sequences, ppls

def sampling_pipeline(model, batch_size, n_samples=1000, top_k=100, max_new_tokens=200, sampling_seed = '<|endoftext|>\nM'):
    prompts = [
        sampling_seed
        for _ in range(n_samples)
    ]

    generated_sequences = model.generate_de_novo(prompts, batch_size=batch_size, max_new_tokens=max_new_tokens) #ensure that sequences end

    generated_sequences = [seq for seq in generated_sequences if 15 < len(clean_protgpt2_generation(seq)) < 250] #limit the generated sequences for AF2 prediction (upper bound is for compute resource)

    most_viable, ppls = get_most_viable(model, generated_sequences, top_k, batch_size=batch_size) # get top_k with lowest ppl

    most_viable = [clean_protgpt2_generation(seq) for seq in most_viable] # clean special tokens and endlines

    return most_viable, ppls


def get_toxicity_scores(model, n_samples=1000, top_k=100, batch_size =8, sampling_seed = 'M'):
    '''
    Given a model and an initiator sequence (ej: "M" ), samples generation from the model, filters by the top-k
    sequences that are more biologically plausible and scores the probability of being toxic. 
    '''

    most_viable, ppls = sampling_pipeline(model, batch_size=batch_size, n_samples=n_samples, top_k=top_k, sampling_seed=sampling_seed)

    toxic_prob, non_toxic_prob = score_toxicity(most_viable, batch_size=batch_size) 
    df = pd.DataFrame(
            zip(most_viable, toxic_prob, non_toxic_prob, ppls),
            columns=["sequence", "tox_score", "non_tox score", "ppl"],
        )
    df.to_csv(f"results/{top_k}_baseline_toxicity_scores.csv", index=False)

    avg_toxic_prob = sum(toxic_prob) / len(toxic_prob) if toxic_prob else 0

    return avg_toxic_prob 


def get_protgpt2():
    from utils.models.protgpt2 import ProtGPT2
    return ProtGPT2("nferruz/ProtGPT2")

def main(args):
    model = get_protgpt2()
    avg_toxic = get_toxicity_scores(model, 
                                    batch_size=args.batch_size, 
                                    n_samples=args.n_samples, 
                                    top_k=args.top_k)
    print(f"Average Toxicity: {avg_toxic}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    main(args)