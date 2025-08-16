from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float


class gPLM(ABC):
    def __init__(self, model_path:str):
        self.model_name_or_path = model_path
        self.model: AutoModelForCausalLM = self._load_model(model_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_path)
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_eoi_str(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass
    
    def generate_de_novo(self, dataset: list[str], batch_size=8, max_new_tokens=100):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        eoi= self._get_eoi_str() + '\n'

        dataset = [eoi+seq for seq in dataset]

        completions = []

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_sequences = self.tokenize_instructions_fn(instructions=dataset[i:i + batch_size])

            generation_toks = self.model.generate(
                input_ids=tokenized_sequences.input_ids.to(self.model.device),
                attention_mask=tokenized_sequences.attention_mask.to(self.model.device),
                generation_config=generation_config,
            )

            generation_toks = generation_toks[:, tokenized_sequences.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                generated = self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                generated = remove_endlines(generated)
                completions.append(
                    generated
                )

        return completions
        
def remove_endlines(text: str, replacement: str = "") -> str:
    """Replace '\n' with `replacement` (default: trim)."""
    return text.replace("\n", replacement)