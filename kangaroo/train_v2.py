from typing import List, Optional, Tuple

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
import glob
import json

import torch
torch.set_default_dtype(torch.bfloat16)

from torch.utils.data import Dataset, DataLoader
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaSdpaAttention,
)
from sklearn.model_selection import train_test_split


@dataclass
class AdapterMode:
    attention_only_mode: str = "attention_only"
    mlp_only_mode: str = "mlp_only"
    decoder_layer_mode: str = "decoder_layer"


class CustomDataset(Dataset):
    def __init__(self, data_dir: str, file_ids: List[int]) -> None:
        self.data = {}
        for data_idx, file_id in enumerate(file_ids):
            self.data[data_idx] = {
                "shallow": os.path.join(data_dir, f"shallow_{file_id}.pt"),
                "target": os.path.join(data_dir, f"target_{file_id}.pt"),
            }

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        shallow_hidden_states = torch.load(self.data[index]["shallow"]).squeeze(0)
        target_hidden_states = torch.load(self.data[index]["target"]).squeeze(0)

        return shallow_hidden_states, target_hidden_states


class KangarooPartialModel(torch.nn.Module):
    def __init__(
        self,
        base_model: str = "../models/meta-llama--Meta-Llama-3.1-8B-Instruct/",
        adapter_mode: str = AdapterMode.decoder_layer_mode,
        lm_head_file: str = "/mnt/kangaroo_train_data/llama-3.1-8b_train_data/lm_head.pt",
        norm_file: str = "/mnt/kangaroo_train_data/llama-3.1-8b_train_data/norm.pt",
    ) -> None:
        super().__init__()

        # Config
        self.draft_temperature = 1.0
        self.target_temperature = 1.0
        self.alpha = 0.8
        self.config = LlamaConfig.from_pretrained(base_model)

        # Init `adapter`
        self.draft_mode_adapter_layer = None
        self.adapter_mode = adapter_mode

        if self.adapter_mode == AdapterMode.attention_only_mode:
            self.attn_input_norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            self.attn_output_norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            self.adapter = LlamaSdpaAttention(
                config=self.config,
                layer_idx=self.config.num_hidden_layers,
            )
        elif self.adapter_mode == AdapterMode.decoder_layer_mode:
            self.adapter = LlamaDecoderLayer(
                config=self.config,
                layer_idx=self.config.num_hidden_layers,
            )

        # Init `lm_head`
        lm_head_state_dict = torch.load(lm_head_file)
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.load_state_dict(lm_head_state_dict)

        # Load norm layer
        norm_state_dict = torch.load(norm_file)
        self.norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.norm.load_state_dict(norm_state_dict)

        # Freeze norm
        for param in self.norm.parameters():
            param.requires_grad = False

        # Freeze lm_head
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def save_adapter(
        self,
        save_dir: str,
        train_loss_history: List[float],
        eval_loss_history: List[float],
    ) -> None:
        """
        Save the parameters of the draft_mode_adapter_layer, loss_history, and shallow_layer_num to the specified directory.

        Args:
            save_dir (str): Directory to save the adapter parameters.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the adapter's state dictionary
        adapter_path = os.path.join(save_dir, "draft_adapter.pt")
        torch.save(self.adapter.state_dict(), adapter_path)
        print(f"Draft adapter saved at {adapter_path}")

        if self.adapter_mode == AdapterMode.attention_only_mode:
            adapter_path = os.path.join(save_dir, "input_norm.pt")
            torch.save(self.attn_input_norm.state_dict(), adapter_path)
            print(f"Draft attn_input_norm saved at {adapter_path}")

            adapter_path = os.path.join(save_dir, "output_norm.pt")
            torch.save(self.attn_output_norm.state_dict(), adapter_path)
            print(f"Draft attn_output_norm saved at {adapter_path}")

        # Save additional information (loss_history and shallow_layer_num)
        metadata = {
            "train_loss_history": train_loss_history,
            "eval_loss_history": eval_loss_history,
        }

        metadata_path = os.path.join(save_dir, "adapter_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        print(f"Adapter metadata saved at {metadata_path}")
    
    def forward(
        self,
        shallow_hidden_states: torch.FloatTensor,
        target_hidden_states: torch.FloatTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            cache_position = torch.arange(
                start=0,
                end=shallow_hidden_states.shape[1],
                device=shallow_hidden_states.device,
            )
            position_ids = cache_position.unsqueeze(0)

        if self.adapter_mode == AdapterMode.attention_only_mode:
            residual = shallow_hidden_states
            shallow_hidden_states = self.attn_input_norm(shallow_hidden_states)
            
            hidden_states, _, _ = self.adapter(
                hidden_states=shallow_hidden_states,
                position_ids=position_ids,
            )
            hidden_states = residual + shallow_hidden_states
            shallow_hidden_states = self.attn_output_norm(shallow_hidden_states)

        elif self.adapter_mode == AdapterMode.decoder_layer_mode:
            layer_outputs = self.adapter(
                hidden_states=shallow_hidden_states,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        draft_logits = self.lm_head(hidden_states) / self.draft_temperature
        target_logits = self.lm_head(target_hidden_states) / self.target_temperature

        # Compute the log probabilities for both models
        draft_log_probs = torch.nn.functional.log_softmax(draft_logits, dim=-1)
        target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

        # Cross-entropy loss between target and draft model predictions
        # kl_loss = torch.nn.functional.kl_div(draft_log_probs, target_probs, reduction="batchmean")
        hard_labels = torch.argmax(target_probs, dim=-1)
        soft_label_cross_entropy_loss = -(target_probs * draft_log_probs).sum(dim=-1).mean()
        hard_label_loss = torch.nn.functional.cross_entropy(
            draft_logits.view(-1, draft_logits.size(-1)),  # Flatten logits
            hard_labels.view(-1)  # Flatten hard labels
        )
        loss = self.alpha * soft_label_cross_entropy_loss + (1 - self.alpha) * hard_label_loss

        return loss


def count_shallow_files(data_dir: str = "/mnt/kangaroo_train_data/llama-3.1-8b_train_data/") -> int:
    shallow_files = glob.glob(os.path.join(data_dir, "shallow_*.pt"))
    return len(shallow_files)


def train(args) -> None:
    # Settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    # Load model
    model = KangarooPartialModel(
        base_model=args.base_model,
        adapter_mode=args.adapter_mode,
        lm_head_file=args.lm_head_file,
        norm_file=args.norm_file,
    ).to(device)
    model = model.to(dtype=torch.bfloat16)

    # Load optimizer
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=lr, foreach=False)
    
    # Load dataset
    ids = range(1, count_shallow_files(data_dir=data_dir) + 1)
    train_ids, eval_ids = train_test_split(ids, test_size=0.1, random_state=2999)

    train_dataset = CustomDataset(data_dir=data_dir, file_ids=train_ids)
    eval_dataset = CustomDataset(data_dir=data_dir, file_ids=eval_ids)

    # Load dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # `position_ids`
    cache_position = torch.arange(
        start=0,
        end=512,
        dtype=torch.bfloat16,
        device=device,
    )
    position_ids = cache_position.unsqueeze(0)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loss_history = []
        eval_loss_history = []
        
        for batch_idx, batch in enumerate(train_dataloader, 1):
            # Zero gradients
            optimizer.zero_grad()

            # Forward propagation
            shallow_hidden_states, target_hidden_states = batch
            shallow_hidden_states = shallow_hidden_states.to(device)
            target_hidden_states = target_hidden_states.to(device)

            loss = model(
                shallow_hidden_states=shallow_hidden_states,
                target_hidden_states=target_hidden_states,
                position_ids=position_ids,
            )

            # Calculate loss
            total_loss += loss.item()
            train_loss_history.append(loss.item())

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Log training loss
            avg_loss = total_loss / batch_idx
            print(f"Train - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(train_dataloader)}], Training Loss: {avg_loss:.4f}")

        # Evaluate the model
        model.eval()
        eval_loss = 0
    
        for batch_idx, batch in enumerate(eval_dataloader, 1):
            # Forward propagation
            shallow_hidden_states, target_hidden_states = batch
            shallow_hidden_states = shallow_hidden_states.to(device)
            target_hidden_states = target_hidden_states.to(device)
        
            with torch.no_grad():
                loss = model(
                    shallow_hidden_states=shallow_hidden_states,
                    target_hidden_states=target_hidden_states,
                )
    
            eval_loss += loss.item()
            eval_loss_history.append(loss.item())

            avg_loss = eval_loss / batch_idx
            print(f"Eval - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(eval_dataloader)}], Eval Loss: {avg_loss:.4f}")

        # Save model checkpoint
        save_dir = "./checkpoints/checkpoints_ce_decoder_layer_20241204/"
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
        model.save_adapter(
            save_path,
            train_loss_history=train_loss_history,
            eval_loss_history=eval_loss_history,
        )
        print(f"Adapter checkpoint saved at {save_path}")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="../models/meta-llama--Meta-Llama-3.1-8B-Instruct/", help="Path to the base model")
    parser.add_argument("--adapter_mode", type=str, default=AdapterMode.decoder_layer_mode, help="Adapter mode")
    parser.add_argument("--lm_head_file", type=str, default="/mnt/kangaroo_train_data/llama-3.1-8b_train_data/lm_head.pt", help="Path to the lm_head file")
    parser.add_argument("--norm_file", type=str, default="/mnt/kangaroo_train_data/llama-3.1-8b_train_data/norm.pt", help="Path to the norm file")
    parser.add_argument("--data_dir", type=str, default="/mnt/kangaroo_train_data/llama-3.1-8b_train_data/", help="Path to the training data directory")
    parser.add_argument("--save_dir", type=str, default=f"./checkpoints/checkpoints_ce_decoder_layer_{datetime.now().strftime('%Y%m%d')}/", help="Path to save the checkpoints")
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha for soft-hard loss ratio")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of dataset used for evaluation")

    args = parser.parse_args()
    train(args)
