from typing import Any, List, Optional, Tuple, Union

from dataclasses import dataclass

import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import Cache, LlamaSdpaAttention
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class KangarooModelMode:
    draft_only_mode: str = "draft_only"
    target_only_model: str = "target_only"
    accelerate_mode: str = "accelerate"


class KangarooGemma2ForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.draft_mode_adapter_layer = LlamaSdpaAttention()(config=config, layer_idx=-1)
        self.shallow_layers = None
        self.mode = KangarooModelMode.target_only_model
    
    def set_skip_layer(self, shallow_layer_num: int) -> None:
        self.shallow_layers = self.model.layers[:shallow_layer_num]
        self.remaining_layers = self.model.layers[shallow_layer_num:]

    def set_draft_mode(self) -> None:
        self.mode = KangarooModelMode.draft_only_mode

    def set_target_mode(self) -> None:
        self.mode = KangarooModelMode.target_only_model

    def set_acceleration_mode(self) -> None:
        self.mode = KangarooModelMode.accelerate_mode

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
