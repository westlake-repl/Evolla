import dataclasses
import json
import os
import random
import types
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

from torch.optim.lr_scheduler import StepLR
from transformers import (AutoTokenizer, BitsAndBytesConfig, LlamaConfig,
                          LlamaForCausalLM)
from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                           QuestionAnsweringModelOutput,
                                           SequenceClassifierOutputWithPast)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward,
                                is_flash_attn_2_available,
                                is_flash_attn_greater_or_equal_2_10, logging,
                                replace_return_docstrings)

from .injection_module import CrossAttention
# from .llama.modeling_llama import LlamaForCausalLM, LlamaModel
from .llm_interface import register_llm
from transformers import AutoConfig

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


_CONFIG_FOR_DOC = "LlamaConfig"
LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


def add_adapter_for_LlamaForCausalLM(llama_for_causalLM, cross_attention_config, num_add_layers=8):
    # follow the same config as the original model
    if num_add_layers < 1:
        return llama_for_causalLM
    cross_attention_config["num_attention_heads"] = llama_for_causalLM.model.layers[0].self_attn.num_heads
    cross_attention_config["hidden_size"] = llama_for_causalLM.model.layers[0].self_attn.hidden_size

    num_layers = len(llama_for_causalLM.model.layers)
    every_n_layers = max(num_layers // num_add_layers, 1)
    # add adapter for each decoder layer
    for i, layer in enumerate(llama_for_causalLM.model.layers):
        if (i + 1) % every_n_layers == 0:
            llama_for_causalLM.model.layers[i].adapter = CrossAttention(**cross_attention_config)

    return llama_for_causalLM


def bind_forward_for_llama(llama_for_causalLM):
    """Bind `forward` function for llama models by `types.MethodType`"""

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def llama_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        protein_feats: Optional[torch.FloatTensor] = None,
        structure_feats: Optional[torch.FloatTensor] = None,
        msa_feats: Optional[torch.FloatTensor] = None,
        protein_batch_mask: Optional[torch.Tensor] = None,
        structure_batch_mask: Optional[torch.Tensor] = None,
        msa_batch_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(
            past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if not hasattr(decoder_layer, 'adapter'):
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                    # keep the hidden_states only, cache other outputs
                    hidden_states = layer_outputs[0]
                    other_outputs = layer_outputs[1:]
                    hidden_states = decoder_layer.adapter(
                        query_states=hidden_states,
                        protein_kv_states=protein_feats,
                        structure_kv_states=structure_feats,
                        msa_kv_states=msa_feats,
                        protein_batch_mask=protein_batch_mask,
                        structure_batch_mask=structure_batch_mask,
                        msa_batch_mask=msa_batch_mask,
                        query_attn_mask=attention_mask,
                    )
                    layer_outputs = (hidden_states,) + other_outputs
            else:
                if not hasattr(decoder_layer, 'adapter'):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )

                    # keep the hidden_states only, cache other outputs
                    hidden_states = layer_outputs[0]
                    other_outputs = layer_outputs[1:]
                    hidden_states = decoder_layer.adapter(
                        query_states=hidden_states,
                        protein_kv_states=protein_feats,
                        structure_kv_states=structure_feats,
                        msa_kv_states=msa_feats,
                        protein_batch_mask=protein_batch_mask,
                        structure_batch_mask=structure_batch_mask,
                        msa_batch_mask=msa_batch_mask,
                        query_attn_mask=attention_mask,
                    )
                    layer_outputs = (hidden_states,) + other_outputs

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def llama_for_causalLM_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        protein_feats: Optional[torch.FloatTensor] = None,
        structure_feats: Optional[torch.FloatTensor] = None,
        msa_feats: Optional[torch.FloatTensor] = None,
        protein_batch_mask: Optional[torch.Tensor] = None,
        structure_batch_mask: Optional[torch.Tensor] = None,
        msa_batch_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            protein_feats=protein_feats,
            structure_feats=structure_feats,
            msa_feats=msa_feats,
            protein_batch_mask=protein_batch_mask,
            structure_batch_mask=structure_batch_mask,
            msa_batch_mask=msa_batch_mask,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    llama_for_causalLM.model.forward = types.MethodType(
        llama_model_forward, llama_for_causalLM.model
    )
    llama_for_causalLM.forward = types.MethodType(
        llama_for_causalLM_forward, llama_for_causalLM
    )

    return llama_for_causalLM


def add_special_tokens_to_model_and_tokenizer(model, tokenizer, special_token):
    # add special tokens to tokenizer # 50265
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    # add special tokens to model # 50272
    if len(tokenizer) <= model.model.embed_tokens.weight.shape[0]:
        return model, tokenizer
    else:
        embedding_layer = model.model.embed_tokens
        embedding_layer.weight.data = torch.cat(
            [
                embedding_layer.weight.data,
                torch.zeros(1, embedding_layer.weight.shape[1]).to(
                    embedding_layer.weight.data
                ),
            ],
            dim=0,
        )
    return model, tokenizer


def bind_function_for_llama(llama_for_causalLM):
    def llama_for_casualLM_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        model_inputs.update(kwargs)
        return model_inputs

    llama_for_causalLM.prepare_inputs_for_generation = types.MethodType(
        llama_for_casualLM_prepare_inputs_for_generation, llama_for_causalLM
    )
    return llama_for_causalLM

from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
from transformers.integrations import HfDeepSpeedConfig

@register_llm
class LlamaAdapterModel(nn.Module):
    def __init__(
        self,
        hf_dir,
        cross_attention_config,
        load_pretrained=True,
        quantization=False,
        attn_implementation="sdpa",
        num_add_layers=8,
    ):
        """Adapter model for Llama.
        Args:
            hf_dir (str): Directory of the Hugging Face model.
            cross_attention_config (dict): Configuration of the cross-attention layer.
            load_pretrained (bool): Whether to load the pretrained model. Defaults to True.
            quantization (bool or str): Whether to use quantization. Defaults to False. Acceptable values are True, False, '8bit', and '4bit'. True means 8-bit quantization. '8bit' means 8-bit quantization. '4bit' means 4-bit quantization.
            attn_implementation (str): Implementation of the attention layer. Defaults to "sdpa".
            num_add_layers (int): Number of additional layers to add. Defaults to 8.
        """
        super().__init__()
        if quantization is True or quantization == '8bit':
            assert load_pretrained, "load_pretrained should be True"
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("8-bit Quantization is enabled")
        elif quantization == '4bit':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            print("4-bit Quantization is enabled")
        else:
            quantization_config = None
            print("Quantization is disabled")
        
        if load_pretrained:
            self.model = LlamaForCausalLM.from_pretrained(
                hf_dir,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            ).train()
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            config = AutoConfig.from_pretrained(hf_dir)
            self.model = LlamaForCausalLM(config)

        self.model = add_adapter_for_LlamaForCausalLM(
            self.model, cross_attention_config, num_add_layers=num_add_layers
        )
        # bind `forward` function for llama models by `types.MethodType`
        self.model = bind_forward_for_llama(self.model)
        self.model = bind_function_for_llama(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_dir, use_fast=False)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"

    def forward(
        self,
        input_ids,
        inputs_mask,
        protein_feats,
        structure_feats,
        msa_feats,
        protein_batch_mask,
        structure_batch_mask,
        msa_batch_mask,
    ):
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=inputs_mask,
            protein_feats=protein_feats,
            structure_feats=structure_feats,
            msa_feats=msa_feats,
            protein_batch_mask=protein_batch_mask,
            structure_batch_mask=structure_batch_mask,
            msa_batch_mask=msa_batch_mask,
            output_hidden_states=True,
        )
        return output

    def generate(
        self,
        input_ids,
        inputs_mask,
        protein_feats,
        structure_feats,
        msa_feats,
        protein_batch_mask,
        structure_batch_mask,
        msa_batch_mask,
        **kwargs
    ):
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        output = self.model.generate(
            input_ids,
            use_cache=False,
            attention_mask=inputs_mask,
            protein_feats=protein_feats,
            structure_feats=structure_feats,
            msa_feats=msa_feats,
            protein_batch_mask=protein_batch_mask,
            structure_batch_mask=structure_batch_mask,
            msa_batch_mask=msa_batch_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=terminators,
            **kwargs,
        )
        output = output[:, input_ids.shape[-1]:]
        return output

    def embed_tokens(self, tokens):
        return self.model.model.embed_tokens(tokens.to(self.model.device))
    
    def generate_prompt(self, question: str) -> str:
        """
        Generate QA prompt for the Llama3-instruct

        Returns: Formatted prompt
        """
        messages = [
            {"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
            {"role": "user", "content": question},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    
    def input_process(self,
                      questions: list,
                      answers: list = None,
                      max_length: int = 512,
                      special_pad_id: int = -100):
        
        # Record original padding side
        original_padding_side = self.tokenizer.padding_side
        
        # Generate prompts for questions
        prompts = [self.generate_prompt(q) for q in questions]

        # Tokenize prompts and add left paddings
        self.tokenizer.padding_side = "left"
        prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )

        input_ids = prompt_inputs["input_ids"]
        attns = prompt_inputs["attention_mask"]
        embeds = self.embed_tokens(input_ids)
        
        # Create labels
        labels = torch.full_like(input_ids, special_pad_id)
        # Create raw text mask
        raw_text_mask = torch.zeros_like(input_ids)

        if answers is not None:
            # Add eos token
            answers_eos = [a + self.tokenizer.eos_token for a in answers]

            # Tokenize answers and add right paddings
            self.tokenizer.padding_side = "right"
            answer_inputs = self.tokenizer(
                answers_eos,
                add_special_tokens=False,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )

            # Concatenate inputs ids
            answer_ids = answer_inputs["input_ids"]
            input_ids = torch.cat([input_ids, answer_ids], dim=-1)

            # Concatenate attention masks
            answer_mask = answer_inputs["attention_mask"]
            attns = torch.cat([attns, answer_mask], dim=-1)

            # Concatenate embeddings
            answer_embeds = self.embed_tokens(answer_ids)
            embeds = torch.cat([embeds, answer_embeds], dim=1)

            # Concatenate labels
            answer_labels = answer_ids.masked_fill(answer_ids == self.tokenizer.pad_token_id, special_pad_id)
            labels = torch.cat([labels, answer_labels], dim=-1)

            # Concatenate raw text mask
            raw_text_mask = torch.cat([raw_text_mask, torch.ones_like(answer_ids)], dim=-1)
            raw_text_mask = raw_text_mask.masked_fill(labels == special_pad_id, 0)

        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, special_pad_id)
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        # Convert to current device
        device = self.model.device
        input_ids = input_ids.to(device)
        embeds = embeds.to(device)
        attns = attns.to(device)
        labels = labels.to(device)
        raw_text_mask = raw_text_mask.to(device)

        return input_ids, embeds, attns, labels, raw_text_mask