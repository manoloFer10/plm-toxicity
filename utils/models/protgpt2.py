import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.models.gPLM import gPLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float


def get_protgpt2():
    protgpt2_model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", torch_dtype="auto", device_map="auto") #738 M params
    protgpt2_tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")

    #important for activations gathering
    protgpt2_tokenizer.padding_side='left'
    protgpt2_tokenizer.pad_token = protgpt2_tokenizer.eos_token

    print('ProtGPT2 model and tokenizer loaded successfully.')
    return protgpt2_model, protgpt2_tokenizer

def get_protgpt2_generation_pipeline():
    protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
    return protgpt2


def add_endlines(sequence):
    endlines = "\n".join(sequence[i:i+60] for i in range(0, len(sequence), 60))
    return "\n" + endlines + "\n"


def format_sequence_protgpt2(sequence: str):
    endlined = add_endlines(sequence)
    return '<|endoftext|>' + endlined + '<|endoftext|>'


def tokenize_sequences_fn(sequences, tokenizer):
    prompts = [
        sequences
    ]

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt"
    )

    return encoded


class ProtGPT2(gPLM):

    def _load_model(self, model_path, dtype= torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model
    
    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return tokenizer
    
    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_sequences_fn, tokenizer=self.tokenizer)

    def _get_eoi_toks(self):
        return self.tokenizer.encode('<|endoftext|>', add_special_tokens=False)
    
    def _get_eoi_str(self):
        return '<|endoftext|>'

    def _get_model_block_modules(self):
        return self.model.transformer.h

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        raise NotImplementedError('orthogonalization matrix')
        return functools.partial(orthogonalize_llama2_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        raise NotImplementedError('add activations on module')
        return functools.partial(act_add_llama2_weights, direction=direction, coeff=coeff, layer=layer)