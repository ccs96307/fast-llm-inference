# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Optional, Tuple, Union

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dataclasses import dataclass
import json

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import dtype
from transformers import (
    LlamaPreTrainedModel,
    GenerationMixin,
)
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from utils.lora import LoRALinear


@dataclass
class MedusaModelMode:
    draft_mode: str = "draft"
    target_mode: str = "target"


class MedusaLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        # Init
        self._mode = MedusaModelMode.target_mode
        self.head_num = None

        self.config = config

        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_medusa_heads(
        self,
        head_num: int = 5,
        torch_dtype: dtype = torch.bfloat16,
        use_lora: bool = False,
        use_low_rank_linear: bool = False,
        share_lm_head_weights: bool = False,
    ) -> None:
        self.head_num = head_num
        if use_lora:
            medusa_heads = [LoRALinear(base_linear=self.lm_head, r=64, torch_dtype=torch_dtype) for _ in range(head_num)]
        elif use_low_rank_linear:
            medusa_heads = [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        self.config.hidden_size,
                        512,
                        bias=False,
                        dtype=torch_dtype,
                    ),
                    torch.nn.Linear(
                        512,
                        self.config.vocab_size,
                        bias=False,
                        dtype=torch_dtype,
                    )
                ) for _ in range(head_num)
            ]
        else:
            medusa_heads = [
                torch.nn.Linear(
                    self.config.hidden_size,
                    self.config.vocab_size,
                    bias=False,
                    dtype=torch_dtype,
                ) for _ in range(head_num)
            ]

        self.medusa_heads = torch.nn.ModuleList(medusa_heads)

        if not use_lora and share_lm_head_weights:
            with torch.no_grad():
                for head in self.medusa_heads:
                    head.weight.copy_(self.lm_head.weight.clone())
                    if self.lm_head.bias is not None:
                        head.bias.copy_(self.lm_head.bias.clone())

    def save_heads(
        self,
        save_dir: str,
        train_loss_history: List[float],
        eval_loss_history: List[float],
    ) -> None:
        """
        Save the parameters of the medusa_heads, loss_history, and head_num to the specified directory.

        Args:
            save_dir (str): Directory to save the heads parameters.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the heads' state dictionary
        heads_path = os.path.join(save_dir, "draft_heads.pt")
        torch.save(self.medusa_heads.state_dict(), heads_path)
        print(f"Medusa heads saved at {heads_path}")

        # Save additional information (loss_history and shallow_layer_num)
        metadata = {
            "train_loss_history": train_loss_history,
            "eval_loss_history": eval_loss_history,
            "head_num": self.head_num
        }

        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        print(f"Medusa metadata saved at {metadata_path}")

    def load_heads(self, load_dir: str) -> None:
        """
        Load the parameters of the medusa_heads, loss_history, and head_num from the specified directory.

        Args:
            load_dir (str): Directory to load the head parameters from.

        Raises:
            FileNotFoundError: If the heads file does not exist in the specified directory.
        """
        heads_path = os.path.join(load_dir, "draft_heads.pt")
        if not os.path.exists(heads_path):
            raise FileNotFoundError(f"Medusa heads not found at {heads_path}")
        
        # Load the adapter's state dictionary
        state_dict = torch.load(heads_path, map_location=self.device)
        self.medusa_heads.load_state_dict(state_dict=state_dict)
        print(f"Medusa heads loaded from {heads_path}")

    def set_draft_mode(self) -> None:
        self._mode = MedusaModelMode.draft_mode

    def set_target_mode(self) -> None:
        self._mode = MedusaModelMode.target_mode

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def get_hidden_states(        
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
        **loss_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        )

        return outputs

    def forward(
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
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.get_hidden_states(
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
        )

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

            if self._mode == MedusaModelMode.draft_mode:
                logits = logits.unsqueeze(0)
                medusa_head_logits = [logits + medusa_head(hidden_states[:, -num_logits_to_keep:, :]).unsqueeze(0) for medusa_head in self.medusa_heads]
                medusa_head_logits = torch.cat(medusa_head_logits)
                logits = torch.cat([logits, medusa_head_logits])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )