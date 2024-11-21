from dataclasses import dataclass



@dataclass
class KangarooModelMode:
    draft_only_mode: str = "draft_only"
    target_only_model: str = "target_only"
    accelerate_mode: str = "accelerate"


