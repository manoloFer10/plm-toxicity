import os
import torch
import json
import functools
from tqdm import tqdm
from utils.scoring import score_toxicity, calculatePerplexity
from utils.extract_activations import add_hooks

def get_most_viable(model, sequences, top_k=100):
    
    tokenizer_fn = model.tokenize_instructions_fn

    scored = [(s, float(calculatePerplexity(s, model, tokenizer_fn))) for s in sequences]
    scored.sort(key=lambda x: x[1])  # lowest perplexity first
    return scored[:max(0, top_k)]


def get_toxicity_scores(model, n_samples=1000, top_k=100, batch_size =8, sampling_seed = 'M'):
    '''
    Given a model and an initiator sequence (ej: "M" ), samples generation from the model, filters by the top-k
    sequences that are more biologically plausible and scores the probability of being toxic. 
    '''

    prompts = [
        sampling_seed
        for _ in n_samples
    ]

    generated_sequences = model.generate_completions(prompts, batch_size=batch_size)

    most_viable = get_most_viable(generated_sequences, top_k)

    toxic_prob, non_toxic_prob = score_toxicity(most_viable) 

    # TODO combine somehow these scores

    return refusal_scores


def select_direction(
    model,
    tokenizer,
    harmful_instructions,
    harmless_instructions,
    candidate_directions,
    artifact_dir,
    kl_threshold=0.1, # directions larger KL score are filtered out
    induce_refusal_threshold=0.0, # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage=0.2, # discard the directions extracted from the last 20% of the model
    batch_size=32
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    baseline_toxicity_scores = get_refusal_scores(model, harmful_instructions, fwd_hooks=[], batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model, harmless_instructions, tokenizer,  fwd_hooks=[], batch_size=batch_size)

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), device=model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), device=model.device, dtype=torch.float64)

    baseline_harmless_logits = get_last_position_logits(
        model=model,
        tokenizer=tokenizer,
        instructions=harmless_instructions,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model.config.num_hidden_layers)]
            fwd_hooks = [(model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.config.num_hidden_layers)]
            fwd_hooks += [(model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.config.num_hidden_layers)]

            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model,
                tokenizer=model.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=tokenizer,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model.config.num_hidden_layers)]
            fwd_hooks = [(model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.config.num_hidden_layers)]
            fwd_hooks += [(model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model.config.num_hidden_layers)]

            refusal_scores = get_refusal_scores(model, harmful_instructions, tokenizer, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal addition for source position {source_pos}"):

            refusal_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(1.0)

            fwd_pre_hooks = [(model_block_modules[source_layer], get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff))]
            fwd_hooks = []

            refusal_scores = get_refusal_scores(model, harmless_instructions, tokenizer, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Ablating direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='ablation_scores'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Adding direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='actadd_scores'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='KL Divergence when ablating direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores'
    )

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
            sorting_score = -refusal_score

            # we filter out directions if the KL threshold 
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['refusal_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")
    
    return pos, layer, candidate_directions[pos, layer]



def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction