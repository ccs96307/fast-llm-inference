from dataclasses import dataclass

from transformers import Gemma2ForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2SdpaAttention


@dataclass
class KangarooModelMode:
    draft_only_mode: str = "draft_only"
    target_only_model: str = "target_only"
    accelerate_mode: str = "accelerate"


class KangarooGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.draft_mode_adapter_layer = Gemma2SdpaAttention()(config=config, layer_idx=-1)
    
    def set_skip_layer(self, shallow_layer_num: int) -> None:
        self.shallow_layers = self.model.layers[:shallow_layer_num]
        self.remaining_layers = self.model.layers[shallow_layer_num:]

    def forward(self, ) -> :