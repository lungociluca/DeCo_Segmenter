import torch
import torch.nn as nn
import os
from src.models.conditioner.base import BaseConditioner

from transformers import Qwen3Model, Qwen2Tokenizer

from config import device

class Qwen3TextEncoder(BaseConditioner):
    def __init__(self, weight_path: str, embed_dim:int=None, max_length=128):
        super().__init__()
        # Normalize the path - remove leading ./ and convert to absolute path if it's a local path
        normalized_path = weight_path.lstrip("./")
        if os.path.exists(normalized_path) or os.path.exists(os.path.abspath(normalized_path)):
            # It's a local path - use the absolute path
            weight_path = os.path.abspath(normalized_path)
        else:
            # It's a HuggingFace model ID - use as is
            weight_path = normalized_path
        
        self.tokenizer = Qwen2Tokenizer.from_pretrained(weight_path, max_length=max_length, padding_side="right")
        # self.model = Qwen3Model.from_pretrained(weight_path, attn_implementation="flex_attention").to(torch.bfloat16)
        self.model = Qwen3Model.from_pretrained(weight_path).to(torch.float32)
        self.model.compile()
        self.uncondition_embedding = None
        self.embed_dim = embed_dim
        self.max_length = max_length
        # torch._dynamo.config.optimize_ddp = False

    def _impl_condition(self, y, metadata:dict={}):
        tokenized = self.tokenizer(y, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

        input_ids = tokenized.input_ids.to(device)
        print(input_ids[:,:13], y)
        # exit(0)
        attention_mask = tokenized.attention_mask.to(device)
        print("tok len", torch.sum(attention_mask))

        metadata["valid_length_y"] = torch.sum(attention_mask, dim=-1)
        attention_mask[:,4:torch.sum(attention_mask).item() - 1] += 10
        y = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if y.shape[2] < self.embed_dim:
            y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], self.embed_dim - y.shape[2]).to(y.device, y.dtype)], dim=-1)
        if y.shape[2] > self.embed_dim:
            y = y[:, :, :self.embed_dim]
        return y, torch.sum(attention_mask) - 1

    def _impl_uncondition(self, y, metadata:dict=None):
        if self.uncondition_embedding is not None and "negative_prompt" not in metadata:
            return self.uncondition_embedding.repeat(len(y), 1, 1)
        negative_prompt = "" if "negative_prompt" not in metadata else metadata['negative_prompt']
        self.uncondition_embedding = self._impl_condition([negative_prompt,])
        return self.uncondition_embedding.repeat(len(y), 1, 1)