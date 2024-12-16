from typing import Dict, List, Optional, Tuple

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from datetime import datetime
import glob
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRMSNorm,
)
from sklearn.model_selection import train_test_split

from utils.lora import LoRALinear


class CustomMedusaDataset(Dataset):
    def __init__(self, data_dir: str, file_ids: List[int], device: torch.DeviceObjType) -> None:
        self.data = {}
        self.device = device
        for data_idx, file_id in enumerate(file_ids):
            self.data[data_idx] = os.path.join(data_dir, f"target_{file_id}.pt")

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor]:
        hidden_states = torch.load(self.data[index]).squeeze(0).to(self.device)
        return hidden_states
    

def collate_fn(batch: List[torch.FloatTensor]) -> torch.FloatTensor:
    """
    Pads the batch of tensors to the same length along the sequence dimension.

    Args:
        batch (List[torch.FloatTensor]): List of tensors from the dataset's __getitem__.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Padded batch tensors and their lengths.
    """
    # Stack tensors into a batch and pad to the maximum length
    # Note: pad_sequence assumes tensors are of shape (seq_len, feature_dim)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return padded_batch


def reshape_logits_with_offset(
    raw_logits: torch.FloatTensor,
    head_num: int,
) -> torch.FloatTensor:
    raw_logits = raw_logits.squeeze(0)
    batch_size, seq_len_with_head_num_padding, vocab_size = raw_logits.shape
    seq_len = seq_len_with_head_num_padding - head_num

    # Init null logits
    updated_logits = torch.zeros((head_num, batch_size, seq_len, vocab_size), device=raw_logits.device)

    # Offset (assume seq_len=512, seq_len_with_head_num_padding=517)
    # head_idx 0: [:, 1:513, :]
    # head_idx 1: [:, 2:514, :]
    # head_idx 2: [:, 3:515, :]
    # head_idx 3: [:, 4:516, :]
    # head_idx 4: [:, 5:517, :]
    for head_idx in range(head_num):
        start_idx = head_idx + 1
        end_idx = seq_len + head_idx + 1
        updated_logits[head_idx] = raw_logits[:, start_idx:end_idx, :]

    return updated_logits


def count_train_files(data_dir: str = "/mnt/cached_train_data/llama-3.1-8b_train_data/") -> int:
    shallow_files = glob.glob(os.path.join(data_dir, "target_*.pt"))
    return len(shallow_files)


class MedusaPartialModel(torch.nn.Module):
    def __init__(
        self,
        base_model: str = "../models/meta-llama--Meta-Llama-3.1-8B-Instruct/",
        lm_head_file: str = "/mnt/cached_train_data/llama-3.1-8b_train_data/lm_head.pt",
        head_num: int = 5,
    ) -> None:
        super().__init__()

        # Config
        self.alpha = 0.2
        self.lambda_k = 0.8
        self.config = LlamaConfig.from_pretrained(base_model)

        # Init `lm_head`
        lm_head_state_dict = torch.load(lm_head_file)
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False, dtype=torch.bfloat16)
        self.lm_head.load_state_dict(lm_head_state_dict)

        # Init `medusa_heads`
        self.head_num = head_num
        self.medusa_heads = torch.nn.ModuleList(
            [
                LoRALinear(
                    base_linear=self.lm_head,
                    r=8,
                    torch_dtype=torch.bfloat16,
                )
                for _ in range(head_num)
            ]
        )

        # Freeze lm_head
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # Unfreeze medusa_head
        for param in self.medusa_heads.parameters():
            param.requires_grad = True

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
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = logits.unsqueeze(0)
        medusa_head_logits = [logits + medusa_head(hidden_states).unsqueeze(0) for medusa_head in self.medusa_heads]
        medusa_head_logits = torch.cat(medusa_head_logits)
        logits = torch.cat([logits, medusa_head_logits])

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        raw_logits = logits[:1, ...]
        heads_logits = logits[1:, :, :-self.head_num, :]

        # Reshape and offset the raw_logits, and the shape must to satisfy the heads_logits
        target_logits = reshape_logits_with_offset(raw_logits=raw_logits, head_num=self.head_num)

        # Compute the log probabilities for both models
        draft_log_probs = torch.nn.functional.log_softmax(heads_logits, dim=-1)
        target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

        # Initialize total loss
        total_loss = 0.0
        for k in range(self.head_num):
            curr_lambda_k = pow(self.lambda_k, k + 1)

            # Extract the k-th head's logits
            k_head_log_probs = draft_log_probs[k]  # Shape: (batch_size, seq_len, vocab_size)
            k_target_logits = target_probs[k]

            # Compute hard labels and soft labels cross entropy loss
            hard_labels = torch.argmax(k_target_logits, dim=-1)
            hard_label_loss = torch.nn.functional.cross_entropy(
                k_head_log_probs.view(-1, draft_log_probs.size(-1)),  # Flatten logits
                hard_labels.view(-1),  # Flatten hard labels
            )
            soft_label_cross_entropy_loss = -(k_target_logits * k_head_log_probs).sum(dim=-1).mean()

            # Combine soft and hard label loss
            head_loss = self.alpha * soft_label_cross_entropy_loss + (1 - self.alpha) * hard_label_loss
            total_loss += curr_lambda_k * head_loss
        
        return total_loss


def train(args) -> None:
    # Settings
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    # Load model
    model = MedusaPartialModel(
        base_model=args.base_model,
        lm_head_file=args.lm_head_file,
    ).to(device)

    # Load optimizer
    optimizer = torch.optim.AdamW(model.medusa_heads.parameters(), lr=lr, foreach=False)
    
    # Load dataset
    ids = range(1, count_train_files(data_dir=data_dir) + 1)
    train_ids, eval_ids = train_test_split(ids, test_size=0.1, random_state=2999)

    train_dataset = CustomMedusaDataset(data_dir=data_dir, file_ids=train_ids, device=device)
    eval_dataset = CustomMedusaDataset(data_dir=data_dir, file_ids=eval_ids, device=device)

    # Load dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
            hidden_states = batch
            loss = model(hidden_states=hidden_states)

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
            hidden_states = batch

            with torch.no_grad():
                loss = model(hidden_states=hidden_states)
    
            eval_loss += loss.item()
            eval_loss_history.append(loss.item())

            avg_loss = eval_loss / batch_idx
            print(f"Eval - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(eval_dataloader)}], Eval Loss: {avg_loss:.4f}")

        # Save model checkpoint
        save_dir = "./checkpoints/checkpoints_ce_decoder_layer_202412015/"
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
        model.save_heads(
            save_path,
            train_loss_history=train_loss_history,
            eval_loss_history=eval_loss_history,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="../models/meta-llama--Meta-Llama-3.1-8B-Instruct/", help="Path to the base model")
    parser.add_argument("--lm_head_file", type=str, default="/mnt/cached_train_data/llama-3.1-8b_train_data/lm_head.pt", help="Path to the lm_head file")
    parser.add_argument("--norm_file", type=str, default="/mnt/cached_train_data/llama-3.1-8b_train_data/norm.pt", help="Path to the norm file")
    parser.add_argument("--data_dir", type=str, default="/mnt/cached_train_data/llama-3.1-8b_train_data/", help="Path to the training data directory")
    parser.add_argument("--save_dir", type=str, default=f"./checkpoints/checkpoints_ce_decoder_layer_{datetime.now().strftime('%Y%m%d')}/", help="Path to save the checkpoints")
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha for soft-hard loss ratio")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of dataset used for evaluation")

    args = parser.parse_args()
    train(args)
