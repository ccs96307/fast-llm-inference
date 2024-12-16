from typing import Dict, List, Optional, Tuple

import os

from datasets import load_dataset
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from medusa_modeling.modeling_medusa_llama import MedusaLlamaForCausalLM


class CustomMedusaDataset(Dataset):
    def __init__(
        self,
        input_data: Dict[str, torch.LongTensor],
    ):
        self.input_data = input_data

    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int):
        return self.input_data[idx]
    

def collate_fn(
    batch: List[str],
    tokenizer: AutoTokenizer,
    device: torch.DeviceObjType,
    head_num: int,
    max_length: int = 512,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    tokenized_batch = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Head num padding
    padding = torch.full(
        (tokenized_batch.input_ids.shape[0], head_num),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_masking = torch.zeros(
        (tokenized_batch.input_ids.shape[0], head_num),
        dtype=torch.long,
        device=device,
    )

    # Concat
    tokenized_batch.input_ids = torch.cat([tokenized_batch.input_ids, padding], dim=1)
    tokenized_batch.attention_mask = torch.cat([tokenized_batch.attention_mask, attention_masking], dim=1)

    return tokenized_batch.input_ids, tokenized_batch.attention_mask


def reshape_logits_with_offset(
    raw_logits: torch.FloatTensor,
    head_num: int,
    raw_attention_mask: torch.LongTensor = None,
) -> Tuple[torch.FloatTensor, Optional[torch.LongTensor]]:
    raw_logits = raw_logits.squeeze(0)
    batch_size, seq_len_with_head_num_padding, vocab_size = raw_logits.shape
    seq_len = seq_len_with_head_num_padding - head_num

    # Init null logits
    updated_logits = torch.zeros((head_num, batch_size, seq_len, vocab_size), device=raw_logits.device)

    # Init updated_attention_mask
    updated_attention_mask = None
    if raw_attention_mask is not None:
        updated_attention_mask = torch.zeros(
            (head_num, batch_size, seq_len),
            device=raw_attention_mask.device,
            dtype=raw_attention_mask.dtype,
        )

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

        if raw_attention_mask is not None:
            updated_attention_mask[head_idx] = raw_attention_mask[:, start_idx:end_idx]

    return updated_logits, updated_attention_mask
    

def main() -> None:
    # Settings
    epochs = 100
    batch_size = 1
    accumulation_steps = 4
    max_length = 256
    lr = 5e-5
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    alpha = 0.2
    head_num = 3
    lambda_k = 0.8

    # Load model and tokenizer
    pretrained_model_name_or_path = "../models/meta-llama--Meta-Llama-3.1-8B-Instruct"
    # pretrained_model_name_or_path = "../models/HuggingFaceTB--SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Teacher model
    model = MedusaLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    model.set_medusa_heads(
        head_num=head_num,
        use_lora=False,
        use_low_rank_linear=True,
        share_lm_head_weights=False,
    )
    model.set_draft_mode()
    model = model.eval().to(device)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze adapter layer
    model.medusa_heads.train()
    for param in model.medusa_heads.parameters():
        param.requires_grad = True

    # Count parameters
    total_params = model.num_parameters()
    print(f"`MedusaModel` has {total_params:,} training parameters.")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"`MedusaModel` has {trainable_params:,} trainable parameters.")

    # Load dataset
    dataset = load_dataset("shibing624/sharegpt_gpt4")
    
    samples = dataset["train"]["conversations"]
    samples = [[{"role": sample[0]["from"].replace("human", "user").replace("gpt", "assistant"), "content": sample[0]["value"]}] for sample in samples]

    train_samples, eval_samples = train_test_split(samples, test_size=0.1, random_state=2999)
    print("Data Size:", len(samples))

    # Tokenized
    train_samples = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in train_samples]
    eval_samples = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in eval_samples]

    train_dataset = CustomMedusaDataset(input_data=train_samples)
    eval_dataset = CustomMedusaDataset(input_data=eval_samples)

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device, head_num=head_num, max_length=max_length),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device, head_num=head_num, max_length=max_length),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print("Start to train.")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loss_history = []
        eval_loss_history = []

        accumulation_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader, 1):
            input_ids, attention_mask = batch

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_logits_to_keep=max_length,
            )

            logits = outputs.logits

            raw_logits = logits[:1, ...]
            heads_logits = logits[1:, :, :-head_num, :]

            # Reshape and offset the raw_logits, and the shape must to satisfy the heads_logits
            target_logits, target_attention_mask = reshape_logits_with_offset(
                raw_logits=raw_logits,
                head_num=head_num,
                raw_attention_mask=attention_mask,
            )

            # Compute the log probabilities for both models
            draft_log_probs = torch.nn.functional.log_softmax(heads_logits, dim=-1)
            target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

            loss = 0.0
            for k in range(head_num):
                curr_lambda_k = pow(lambda_k, k + 1)

                # Extract the k-th head's logits
                k_head_log_probs = draft_log_probs[k]  # Shape: (batch_size, seq_len, vocab_size)
                k_target_probs = target_probs[k]

                # Compute hard labels and soft labels cross entropy loss
                hard_labels = torch.argmax(k_target_probs, dim=-1)
                hard_label_loss = torch.nn.functional.cross_entropy(
                    k_head_log_probs.view(-1, draft_log_probs.size(-1)),  # Flatten logits
                    hard_labels.view(-1),  # Flatten hard labels
                    ignore_index=tokenizer.pad_token_id,
                )

                # Mask padding token
                valid_mask = target_attention_mask[k].bool()
                masked_k_head_log_probs = k_head_log_probs[valid_mask]
                masked_k_target_probs = k_target_probs[valid_mask]

                soft_label_cross_entropy_loss = -(masked_k_target_probs * masked_k_head_log_probs).sum(dim=-1).mean()

                # Combine soft and hard label loss
                head_loss = alpha * soft_label_cross_entropy_loss + (1 - alpha) * hard_label_loss
                loss += curr_lambda_k * head_loss
            
            # Accumulation
            loss = loss / accumulation_steps
            accumulation_loss += loss.item()
            
            # Backward pass
            loss.backward()

            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_dataloader):
                # Optimizer step
                optimizer.step()

                # Zero gradients
                optimizer.zero_grad()

                # Log training loss
                total_loss += accumulation_loss
                train_loss_history.append(accumulation_loss)
                avg_loss = total_loss / (batch_idx / accumulation_steps)
                accumulation_loss = 0.0
                print(f"Train - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(train_dataloader)}], Training Loss: {avg_loss:.4f}")

        # Evaluate the model
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader, 1):
                input_ids, attention_mask = batch

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_logits_to_keep=max_length,
                )

                logits = outputs.logits
                raw_logits = logits[:1, ...]
                heads_logits = logits[1:, :, :-head_num, :]

                # Reshape and offset the raw_logits, and the shape must to satisfy the heads_logits
                target_logits, target_attention_mask = reshape_logits_with_offset(
                    raw_logits=raw_logits,
                    head_num=head_num,
                    raw_attention_mask=attention_mask,
                )

                # Compute the log probabilities for both models
                draft_log_probs = torch.nn.functional.log_softmax(heads_logits, dim=-1)
                target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

                loss = 0.0
                for k in range(head_num):
                    curr_lambda_k = pow(lambda_k, k + 1)

                    # Extract the k-th head's logits
                    k_head_log_probs = draft_log_probs[k]  # Shape: (batch_size, seq_len, vocab_size)
                    k_target_probs = target_probs[k]

                    # Compute hard labels and soft labels cross entropy loss
                    hard_labels = torch.argmax(k_target_probs, dim=-1)
                    hard_label_loss = torch.nn.functional.cross_entropy(
                        k_head_log_probs.view(-1, draft_log_probs.size(-1)),  # Flatten logits
                        hard_labels.view(-1),  # Flatten hard labels
                        ignore_index=tokenizer.pad_token_id,
                    )

                    # Mask padding token
                    valid_mask = target_attention_mask[k].bool()
                    masked_k_head_log_probs = k_head_log_probs[valid_mask]
                    masked_k_target_probs = k_target_probs[valid_mask]

                    soft_label_cross_entropy_loss = -(masked_k_target_probs * masked_k_head_log_probs).sum(dim=-1).mean()

                    # Combine soft and hard label loss
                    head_loss = alpha * soft_label_cross_entropy_loss + (1 - alpha) * hard_label_loss
                    loss += curr_lambda_k * head_loss

                eval_loss += loss.item()
                eval_loss_history.append(loss.item())

                avg_loss = eval_loss / batch_idx
                print(f"Eval - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(eval_dataloader)}], Eval Loss: {avg_loss:.4f}")

        # Save model checkpoint
        save_dir = "./checkpoints/checkpoints_hce_linear_20241217/"
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
        model.save_heads(
            save_path,
            train_loss_history=train_loss_history,
            eval_loss_history=eval_loss_history,
        )
        print(f"checkpoint saved at {save_path}")
    

if __name__ == "__main__":
    main()
