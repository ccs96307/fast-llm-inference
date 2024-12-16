import torch


class LoRALinear(torch.nn.Module):
    def __init__(
        self,
        base_linear: torch.nn.Linear,
        r: float,
        alpha: float = 1.0,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r  # Low-rank dimension
        self.alpha = alpha

        # LoRA
        self.lora_A = torch.nn.Parameter(torch.empty(r, self.in_features, dtype=torch_dtype))
        self.lora_B = torch.nn.Parameter(torch.empty(self.out_features, r, dtype=torch_dtype))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B)

        # Scaling factor
        self.scaling = alpha / min(r, 1.0)

    def forward(self, x):
        _r_hidden_states = torch.nn.functional.linear(x, self.lora_A)
        lora_increment = self.scaling * torch.nn.functional.linear(_r_hidden_states, self.lora_B)
        return lora_increment
