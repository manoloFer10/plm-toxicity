import os
import torch
import json
import math
import gc
import pandas as pd
from tqdm import tqdm
from utils.models.gPLM import gPLM
from utils.scoring import score_toxicity, calculatePerplexity
from utils.models.protgpt2 import clean_protgpt2_generation
from utils.extract_activations import (
    add_hooks, get_activation_addition_input_pre_hook, 
    get_direction_ablation_input_pre_hook, 
    get_direction_ablation_output_hook
    )
from utils.visualizations import plot_tox_scores


def get_most_viable(model, sequences, top_k=100, batch_size = 8):
    
    tokenizer_fn = model.tokenize_instructions_fn

    top_k = min(top_k, len(sequences)) #ensure top_k is not greater than available sequences

    scored = []
    batch_sequences = []
    for seq in sequences:
        batch_sequences.append(seq)
        if len(batch_sequences) == batch_size:
            ppls = calculatePerplexity(batch_sequences, model, tokenizer_fn)
            ppls = ppls.tolist() if hasattr(ppls, "tolist") else list(ppls)
            for s, ppl in zip(batch_sequences, ppls):
                scored.append((s, float(ppl)))
                scored.sort(key=lambda x: x[1])
                if len(scored) > top_k:
                    scored.pop()
            batch_sequences = []

    if batch_sequences:
        ppls = calculatePerplexity(batch_sequences, model, tokenizer_fn)
        ppls = ppls.tolist() if hasattr(ppls, "tolist") else list(ppls)
        for s, ppl in zip(batch_sequences, ppls):
            scored.append((s, float(ppl)))
            scored.sort(key=lambda x: x[1])
            if len(scored) > top_k:
                scored.pop()

    sequences = [s for s, _ in scored]
    ppls = [p for _, p in scored]
    return sequences, ppls

def sampling_pipeline(model, batch_size, n_samples=1000, top_k=100, max_new_tokens=200, sampling_seed = 'M', fwd_pre_hooks=[], fwd_hooks=[]):
    prompts = [
        sampling_seed
        for _ in range(n_samples)
    ]

    generated_sequences = model.generate_de_novo(prompts, 
                                                 batch_size=batch_size, 
                                                 max_new_tokens=max_new_tokens,
                                                 fwd_pre_hooks=fwd_pre_hooks,
                                                 fwd_hooks=fwd_hooks) #generate with hooks

    generated_sequences = [seq for seq in generated_sequences if 15 < len(clean_protgpt2_generation(seq)) < 250] #limit the generated sequences for AF2 prediction (upper bound is for compute resource)

    most_viable, ppls = get_most_viable(model, generated_sequences, top_k, batch_size=32) # get top_k with lowest ppl

    most_viable = [clean_protgpt2_generation(seq) for seq in most_viable] # clean special tokens and endlines

    return most_viable, ppls

def get_toxicity_scores(model, n_samples=1000, top_k=100, batch_size =8, sampling_seed = 'M', artifact_path='generations', setting='default', fwd_pre_hooks=[], fwd_hooks=[]):
    '''
    Given a model and an initiator sequence (ej: "M" ), samples generation from the model, filters by the top-k
    sequences that are more biologically plausible and scores the probability of being toxic. 
    '''

    most_viable, ppls = sampling_pipeline(model, 
                                          batch_size=batch_size, 
                                          n_samples=n_samples, 
                                          top_k=top_k,
                                          sampling_seed=sampling_seed,
                                          fwd_pre_hooks=fwd_pre_hooks,
                                          fwd_hooks=fwd_hooks)

    toxic_prob, non_toxic_prob = score_toxicity(most_viable, batch_size=50) 
    
    df = pd.DataFrame(
            zip(most_viable, toxic_prob, non_toxic_prob, ppls),
            columns=["sequence", "tox_score", "non_tox score", "ppl"],
        )
    df.to_csv(artifact_path + f'/{setting}', index=False)

    avg_toxic_prob = sum(toxic_prob) / len(toxic_prob) if toxic_prob else 0

    print(f'For {setting} got: \n Average Toxicity: {avg_toxic_prob}')
    return avg_toxic_prob
    


def get_last_position_logits(model, instructions, tokenize_instructions_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    last_position_logits = None

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(sequences=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            with torch.no_grad():
                logits = model(
                    input_ids=tokenized_instructions.input_ids.to(model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(model.device),
                ).logits

            if last_position_logits is None:
                last_position_logits = logits[:, -1, :]
            else:
                last_position_logits = torch.cat((last_position_logits, logits[:, -1, :]), dim=0)

    return last_position_logits # -> Float[Tensor, "n_instructions d_vocab"]


def select_direction(
    model: gPLM,
    kl_validation_samples, #fetch these from not used protein sets.
    candidate_directions,
    artifact_dir,
    n_samples:int = 1000,
    top_k:int = 100,
    kl_threshold:float = 0.1, # directions larger KL score are filtered out

    #induce_refusal_threshold=0.0, # directions with a lower inducing refusal score are filtered out MANU: what to do with this

    prune_layer_percentage=0.2, # discard the directions extracted from the last 20% of the model
    batch_size=64,
    layer_step: int = 2
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    print(f'Executing direction selection pipeline with {layer_step=}')

    baseline_tox_score = get_toxicity_scores(model, n_samples, top_k=top_k, artifact_path=artifact_dir, batch_size=batch_size, setting='baseline')

    ablation_kl_div_scores = torch.full((n_pos, n_layer), float("nan"), device=model.device, dtype=torch.float64)
    ablation_tox_scores = torch.full((n_pos, n_layer), float("nan"), device=model.device, dtype=torch.float64)
    steering_tox_scores = torch.full((n_pos, n_layer), float("nan"), device=model.device, dtype=torch.float64)

    baseline_non_tox_logits = get_last_position_logits(
        model=model.model,
        instructions=kl_validation_samples,
        tokenize_instructions_fn=model.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    ) #

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(0, n_layer, layer_step), desc=f"Computing KL for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model.model.config.num_hidden_layers)]
            fwd_hooks = [(model.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.model.config.num_hidden_layers)]
            fwd_hooks += [(model.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.model.config.num_hidden_layers)]

            intervention_logits = get_last_position_logits(
                model=model.model,
                instructions=kl_validation_samples,
                tokenize_instructions_fn=model.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            ) # : Float[Tensor, "n_instructions 1 d_vocab"]

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_non_tox_logits, intervention_logits).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(0, n_layer, layer_step), desc=f"Computing refusal ablation for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model.model.config.num_hidden_layers)]
            fwd_hooks = [(model.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.model.config.num_hidden_layers)]
            fwd_hooks += [(model.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.model.config.num_hidden_layers)]

            ablation_tox_score = get_toxicity_scores(model, 
                                             n_samples, 
                                             top_k=top_k, 
                                             batch_size=batch_size, 
                                             artifact_path=artifact_dir, 
                                             setting= f'l{source_layer}_pos{source_pos}_toxicity_ablation', 
                                             fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks
            )
            ablation_tox_scores[source_pos, source_layer] = ablation_tox_score

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(0, n_layer, layer_step), desc=f"Computing refusal addition for source position {source_pos}"):

            tox_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(1.0)

            fwd_pre_hooks = [(model.model_block_modules[source_layer], get_activation_addition_input_pre_hook(vector=tox_vector, coeff=coeff))]
            fwd_hooks = []

            steering_tox_score = get_toxicity_scores(model, 
                                                 n_samples, 
                                                 top_k=top_k, 
                                                 batch_size=batch_size, 
                                                 artifact_path=artifact_dir, 
                                                 setting= f'l{source_layer}_pos{source_pos}_toxicity_addition', 
                                                 fwd_pre_hooks=fwd_pre_hooks, 
                                                 fwd_hooks=fwd_hooks)
            steering_tox_scores[source_pos, source_layer] = steering_tox_score

    try:
        plot_tox_scores(
            tox_scores=ablation_tox_scores,
            baseline_tox_score=baseline_tox_score,
            title='Ablating direction',
            artifact_dir=artifact_dir,
            artifact_name='ablation_scores'
        )

        plot_tox_scores(
            tox_scores=steering_tox_scores,
            baseline_tox_score=baseline_tox_score,
            title='Adding direction',
            artifact_dir=artifact_dir,
            artifact_name='actadd_scores'
        )

        plot_tox_scores(
            tox_scores=ablation_kl_div_scores,
            baseline_tox_score=0.0,
            title='KL Divergence when ablating direction on non-toxic sequences',
            artifact_dir=artifact_dir,
            artifact_name='kl_div_scores'
        )
    except Exception as e: 
        print(f'Error when creating figures: {e}')

    def _json_sanitize(x: float):
        return None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            ablation_tox_score = float(ablation_tox_scores[source_pos, source_layer].item())
            steering_tox_score = float(steering_tox_scores[source_pos, source_layer].item())
            kl_div_score = float(ablation_kl_div_scores[source_pos, source_layer].item())

            json_ablation_tox_score = _json_sanitize(ablation_tox_score)
            json_steering_tox_score = _json_sanitize(steering_tox_score)
            json_kl_div_score = _json_sanitize(kl_div_score)

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'ablation_tox_score': json_ablation_tox_score,
                'steering_score': json_steering_tox_score,
                'kl_div_score': json_kl_div_score
            })


            # we filter out directions if the KL threshold 
            discard_direction = filter_fn(
                ablation_tox_score=ablation_tox_score,
                steering_tox_score=steering_tox_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            # select the direction that maximizes toxicity
            filtered_scores.append((steering_tox_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'ablation_tox_score': ablation_tox_score,
                'steering_score': steering_tox_score,
                'kl_div_score': kl_div_score
            })

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['ablation_tox_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Ablation Tox score: {ablation_tox_scores[pos, layer].item():.4f} (baseline: {baseline_tox_score:.4f})")
    print(f"Steering Tox score: {steering_tox_scores[pos, layer].item():.4f} (baseline: {baseline_tox_score:.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer].item():.4f}")

    return pos, layer, candidate_directions[pos, layer]



def select_and_save_direction(artifact_dir, model_base, kl_validation_samples, candidate_directions, n_samples = 1000, top_k = 100, batch_size=128):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(artifact_dir, 'select_direction')):
        os.makedirs(os.path.join(artifact_dir, 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        kl_validation_samples,
        candidate_directions,
        artifact_dir=os.path.join(artifact_dir, "select_direction"),
        n_samples = n_samples,
        top_k = top_k,
        batch_size = batch_size
    )

    with open(f'{artifact_dir}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{artifact_dir}/direction.pt')

    return pos, layer, direction


def kl_div_fn(
    logits_a, #: Float[Tensor, 'batch seq_pos d_vocab']
    logits_b, #: Float[Tensor, 'batch seq_pos d_vocab']
    epsilon=1e-6 #: Float=1e-6
): # -> Float[Tensor, 'batch']
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(
        probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), 
        dim=-1
    )

    return torch.mean(kl_divs, dim=-1)
    

    # returns True if the direction should be filtered out
def filter_fn(ablation_tox_score, steering_tox_score, kl_div_score, layer, n_layer, kl_threshold=None, baseline_tox_score=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(ablation_tox_score) or math.isnan(steering_tox_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if baseline_tox_score is not None and ablation_tox_score >= baseline_tox_score: #ablation is generating more toxic proteins
        return True
    if baseline_tox_score is not None and steering_tox_score <= baseline_tox_score: #steering is generating less toxic proteins
        return True
    return False