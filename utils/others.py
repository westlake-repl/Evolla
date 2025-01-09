import os
import random
import re
import time

import numpy as np
import torch
# from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import yaml
from utils.easydict import MyEasyDict
from model.model_interface import ModelInterface

structure_encoder_name_2_protein_dim = {
    "SaProt_35M_AF2": 480,
    "SaProt_650M_AF2": 1280,
}

protein_encoder_name_2_protein_dim = {
    "esm2_t12_35M_UR50D": 480,
    "esm2_t33_650M_UR50D": 1280,
    "SaProt_35M_AF2": 480,
    "SaProt_650M_AF2": 1280,
    "ProTrek_35M_seq": 480,
    "ProTrek_650M_seq": 1280,
}

llm_name_2_llm_embedding_dim = {
    "opt-350m": 512,
    "facebook-opt-350m": 512,
    "meta-llama_Meta-Llama-3-8B": 4096,
    "meta-llama_Meta-Llama-3-8B-Instruct": 4096,
    "opt-2.7b": 2560,
    "Qwen1.5-0.5B": 1024,
    "Qwen1.5-4B-Chat": 1024,
    "phi-1_5": 2048,
    "phi-2": 2560,
    "Llama2hf7b": 4096,
}

def setup_seed(seed):
    """set random seed for reproducibility.
    Args:
        seed (int): random seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def align_model_config(config: MyEasyDict):
    """Align model config. Different model sometimes should share the same dimension, but it's not easy to set them manually.
    Args:
        config (MyEasyDict): model config.
    
    Returns:
        config (MyEasyDict): aligned model config.
    """

    # if config.fusion_module.output_repr_dim is not set, it should be same as llm embedding dim
    llm_name = config.llm.hf_dir.split("/")[-1]  # example: opt-350m
    llm_embedding_dim = llm_name_2_llm_embedding_dim[llm_name]

    if config.protein_encoder is not None:
        # get protein dim by protein_encoder.config_path
        protein_encoder_name = config.protein_encoder.config_path.split("/")[
            -1
        ]  # example: esm2_t12_35M_UR50D
        protein_encoder_dim = protein_encoder_name_2_protein_dim[protein_encoder_name]
        # assign protein_encoder_dim to config.protein_encoder.fusion_module.protein_repr_dim
        # config.fusion_module.protein_repr_dim = protein_encoder_dim
        config.protein_encoder.fusion_module.protein_repr_dim = protein_encoder_dim
        # config.fusion_module.output_repr_dim = llm_embedding_dim
        if config.protein_encoder.fusion_module.output_repr_dim is None:
            config.protein_encoder.fusion_module.output_repr_dim = llm_embedding_dim

        # align config.llm.cross_attention_config.encoder_dim with config.fusion_module.output_repr_dim
        if config.llm.get("cross_attention_config", None) is not None:
            # config.llm.cross_attention_config.encoder_dim = config.fusion_module.output_repr_dim
            config.llm.cross_attention_config.protein_encoder_dim = (
                config.protein_encoder.fusion_module.output_repr_dim
            )

    if config.structure_encoder is not None:
        if "config_path" in config.structure_encoder:  # for saprot
            structure_encoder_name = config.structure_encoder.config_path.split("/")[-1]
        elif "tokenizer_path" in config.structure_encoder:  # for structure embedding
            structure_encoder_name = config.structure_encoder.tokenizer_path.split("/")[
                -1
            ]
        else:  # for GNN
            structure_encoder_name = None
        if structure_encoder_name is not None:
            structure_encoder_dim = structure_encoder_name_2_protein_dim[
                structure_encoder_name
            ]
        else:
            structure_encoder_dim = 512  # TODO

        if "fusion_module" in config.structure_encoder:
            config.structure_encoder.fusion_module.protein_repr_dim = (
                structure_encoder_dim
            )

            if config.structure_encoder.fusion_module.output_repr_dim is None:
                config.structure_encoder.fusion_module.output_repr_dim = (
                    llm_embedding_dim
                )

        # align config.llm.cross_attention_config.encoder_dim with config.fusion_module.output_repr_dim
        if config.llm.get("cross_attention_config", None) is not None:
            if "fusion_module" in config.structure_encoder:
                config.llm.cross_attention_config.structure_encoder_dim = (
                    config.structure_encoder.fusion_module.output_repr_dim
                )
            else:
                config.llm.cross_attention_config.structure_encoder_dim = (
                    structure_encoder_dim
                )

    if config.msa_encoder is not None:
        msa_encoder_dim = 768

        if "fusion_module" in config.msa_encoder:
            config.msa_encoder.fusion_module.protein_repr_dim = msa_encoder_dim

            if config.msa_encoder.fusion_module.output_repr_dim is None:
                config.msa_encoder.fusion_module.output_repr_dim = llm_embedding_dim

        # align config.llm.cross_attention_config.encoder_dim with config.fusion_module.output_repr_dim
        if config.llm.get("cross_attention_config", None) is not None:
            if "fusion_module" in config.msa_encoder:
                config.llm.cross_attention_config.msa_encoder_dim = (
                    config.msa_encoder.fusion_module.output_repr_dim
                )
            else:
                config.llm.cross_attention_config.msa_encoder_dim = msa_encoder_dim

    return config


def filter_llama_weights(state_dict):
    """Filter out llama weights from state_dict because of training issues. The llama weights have already been loaded while initializing the model."""
    llama_keys = []
    for k, v in state_dict.items():
        if k.startswith("llm.") and 'adapter' not in k:
            llama_keys.append(k)
        if k.startswith("model.3.") and 'adapter' not in k:
            llama_keys.append(k)
    for k in llama_keys:
        state_dict.pop(k)
    return state_dict


def get_prompt(sequence, structure, question):
    """Generate prompt and SA sequence for SaProt.
    
    Args:
        sequence (str): amino acid sequence.
        structure (str): structure sequence represented by foldseek.
        question (str): question for the model.
    
    Returns:
        prompt (str): prompt for the model.
        sequence (str): sequence with structure information.
    """
    sequence_template = "<SaProt_seq><SaProtRepr_seq></SaProt_seq><sep>Question: {Question} Answer: "
    structure_template = "<SaProt_struct><SaProtRepr_struct></SaProt_struct><sep>Question: {Question} Answer: "
    saprot_template = "<SaProt><SaProtRepr></SaProt><sep>Question: {Question} Answer: "
    if sequence is not None and structure is not None:
        if len(sequence) != len(structure):
            raise ValueError(f"The length of sequence and structure are not equal. {len(sequence)}!= {len(structure)}")
        _sequence = sequence.upper()
        _structure = structure.lower()
        sequence = "".join([f"{_seq}{_struct}" for _seq, _struct in zip(_sequence, _structure)])
        print("all", sequence)
        prompt = saprot_template.format(Question=question)
    elif sequence is not None:
        _sequence = sequence.upper()
        _structure = "#" * len(_sequence)
        sequence = "".join([f"{_seq}{_struct}" for _seq, _struct in zip(_sequence, _structure)])
        print("seqonly", sequence)
        prompt = sequence_template.format(Question=question)
    elif structure is not None:
        _sequence = "#" * len(structure)
        _structure = structure.lower()
        sequence = "".join([f"{_seq}{_struct}" for _seq, _struct in zip(_sequence, _structure)])
        prompt = structure_template.format(Question=question)
        print("structonly", sequence)
    return prompt, sequence



def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as r:
        config = MyEasyDict(yaml.safe_load(r))
    config.model.config = align_model_config(config.model.config)
    return config

def load_model_from_config(config, local_rank=0, dtype=None):
    """load model from config.
    Args:
        config (MyEasyDict): config of the model.
        local_rank (int): local rank of the current process.
        dtype (str): data type of the model. Default is None. Options are "fp32", "fp16", "bf16".
    
    Returns:
        model (nn.Module): loaded model.
    """
    model_py_path = config.model.pop("cls")
    model = ModelInterface.init_model(model_py_path, **config.model)
    model.eval()

    ckpt = torch.load(os.path.join(config.setting.from_checkpoint, "checkpoint", "mp_rank_00_model_states.pt"), map_location=f'cpu')
    state_dict = ckpt["module"]
    state_dict = filter_llama_weights(state_dict)
    model.load_state_dict(state_dict, strict=False)
    if dtype is None:
        pass
    elif dtype == "fp32":
        model.to(torch.float32)
    elif dtype == "bf16":
        model.to(torch.bfloat16)
    elif dtype == "fp16":
        model.to(torch.float16)
    else:
        raise ValueError(f"Unsupported data type: {dtype}, supported data types are 'fp32', 'fp16', 'bf16'")
    model.to(f'cuda:{local_rank}')
    return model