from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def get_protgpt2():
    protgpt2_model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", torch_dtype="auto", device_map="auto") #738 M params
    protgpt2_tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")

    #important for activations gathering
    protgpt2_tokenizer.padding_side='left'
    protgpt2_tokenizer.pad_token = protgpt2_tokenizer.eos_token

    return protgpt2_model, protgpt2_tokenizer
