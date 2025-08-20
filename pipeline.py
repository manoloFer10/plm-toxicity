import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_dataset
from utils.models.protgpt2 import ProtGPT2
from utils.extract_activations import generate_directions
from utils.select_steering import select_and_save_direction

def get_model():
    return ProtGPT2("nferruz/ProtGPT2")

def get_activation_extraction_data():
    data = load_dataset('manufernandezbur/balanced_toxfeatures')['train']

    tox = data.filter(lambda x: x['Toxin']).to_pandas()['Sequence']
    non_tox = data.filter(lambda x: not x['Toxin']).to_pandas()['Sequence']

    return list(tox), list(non_tox)

def main():

    n_samples = 1000
    top_k=50
    batch_size=128


    artifact_dir = 'activations'
    os.makedirs(artifact_dir, exist_ok=True)

    model = get_model()
    tox, non_tox = get_activation_extraction_data()
    kl_validation_samples = [model._get_eoi_str for _ in range(32)] # generate 32 eos initiating tokens.

    mean_diffs = generate_directions(model.model, model.tokenizer, model.model_block_modules, tox, non_tox, artifact_dir)

    pos, layer, direction = select_and_save_direction(
                                        artifact_dir=artifact_dir,
                                        model_base=model,
                                        kl_validation_samples=kl_validation_samples,
                                        candidate_directions=mean_diffs,
                                        n_samples=n_samples,
                                        top_k=top_k,
                                        batch_size=batch_size
                                    )
    
    print(f'Successfully extracted best direction in pos {pos} and layer {layer}: {direction=}')



if __name__ =='__main__':
    main()