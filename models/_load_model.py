# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import MODEL_PATH


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=None):
    if torch_dtype is None:
        torch_dtype = torch.float16 if (torch.cuda.is_available() and 'cuda' in str(device)) else torch.float32

    local_path = os.path.abspath(os.path.join(MODEL_PATH, model_name)).replace('\\', '/')
    if os.path.exists(local_path):
        if model_name.startswith('facebook/opt'):
            model = OPTForCausalLM.from_pretrained(local_path, torch_dtype=torch_dtype)
        elif model_name == "falcon-7b":
            model = AutoModelForCausalLM.from_pretrained(local_path, cache_dir=None, trust_remote_code=True, torch_dtype=torch_dtype, local_files_only=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(local_path, cache_dir=None, torch_dtype=torch_dtype, local_files_only=True)
    elif model_name == "microsoft/deberta-large-mnli" or model_name == 'roberta-large-mnli':
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif model_name.startswith('facebook/opt'):
        opt_local = os.path.abspath(os.path.join(MODEL_PATH, model_name.split("/")[1])).replace('\\', '/')
        if os.path.exists(opt_local):
            model = OPTForCausalLM.from_pretrained(opt_local, torch_dtype=torch_dtype)
        else:
            model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)

    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    local_path = os.path.abspath(os.path.join(MODEL_PATH, model_name)).replace('\\', '/')
    if os.path.exists(local_path):
        if model_name == "falcon-7b":
            tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, cache_dir=None, use_fast=use_fast, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(local_path, cache_dir=None, use_fast=use_fast, local_files_only=True)
            tokenizer.eos_token_id = tokenizer.eos_token_id or 2
            tokenizer.bos_token_id = tokenizer.bos_token_id or 1
            tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
            tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
    elif model_name.startswith('facebook/opt'):
        opt_local = os.path.abspath(os.path.join(MODEL_PATH, model_name.split("/")[1])).replace('\\', '/')
        if os.path.exists(opt_local):
            tokenizer = AutoTokenizer.from_pretrained(opt_local, use_fast=use_fast)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer