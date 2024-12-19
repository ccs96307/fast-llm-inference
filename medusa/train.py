from typing import Dict, List, Optional, Tuple

import os

from datasets import load_dataset, load_from_disk
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
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
    
    def __getitem__(self, idx: int) -> str:
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
        (tokenized_batch.input_ids.shape[0], head_num + 1),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_masking = torch.zeros(
        (tokenized_batch.input_ids.shape[0], head_num + 1),
        dtype=torch.long,
        device=device,
    )

    # Concat
    tokenized_batch.input_ids = torch.cat([tokenized_batch.input_ids, padding], dim=1)
    tokenized_batch.attention_mask = torch.cat([tokenized_batch.attention_mask, attention_masking], dim=1)

    return tokenized_batch.input_ids, tokenized_batch.attention_mask


def reshape_labels_with_offset(
    input_ids: torch.LongTensor,
    head_num: int,
    tokenizer: AutoTokenizer,
    raw_attention_mask: torch.LongTensor = None,
) -> Tuple[torch.FloatTensor, Optional[torch.LongTensor]]:
    # input_ids shape: [batch_size, seq_len_with_head_num_padding]
    batch_size, seq_len_with_head_num_padding = input_ids.shape
    seq_len = seq_len_with_head_num_padding - head_num - 1

    # Init null logits
    updated_labels = torch.full(
        (head_num, batch_size, seq_len),
        tokenizer.pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    # Init updated_attention_mask
    updated_attention_mask = None
    if raw_attention_mask is not None:
        updated_attention_mask = torch.zeros(
            (head_num, batch_size, seq_len),
            device=raw_attention_mask.device,
            dtype=raw_attention_mask.dtype,
        )

    # Offset (assume seq_len = 512 and head_num = 5, seq_len_with_head_num_padding = 512 + 5 + 1 = 518)
    # head_idx 0: [:, 2:514]
    # head_idx 1: [:, 3:515]
    # head_idx 2: [:, 4:516]
    # head_idx 3: [:, 5:517]
    # head_idx 4: [:, 6:518]
    for head_idx in range(head_num):
        start_idx = head_idx + 2
        end_idx = seq_len + head_idx + 2
        updated_labels[head_idx] = input_ids[:, start_idx:end_idx]

        if raw_attention_mask is not None:
            updated_attention_mask[head_idx] = raw_attention_mask[:, start_idx:end_idx]

    return updated_labels, updated_attention_mask
    

def main() -> None:
    # Settings
    epochs = 100
    batch_size = 4
    accumulation_steps = 4
    max_length = 512
    lr = 5e-5
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    head_num = 1
    lambda_k = 1
    print(f"Device: {device}")

    # Load model and tokenizer
    pretrained_model_name_or_path = "../models/meta-llama--Meta-Llama-3.1-8B-Instruct"
    # pretrained_model_name_or_path = "../models/meta-llama--Meta-Llama-3-8B-Instruct"
    # pretrained_model_name_or_path = "../models/HuggingFaceTB--SmolLM2-135M-Instruct"
    pretrained_model_name_or_path = "../models/lmsys--vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Teacher model
    model = MedusaLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    model.set_medusa_heads(
        head_num=head_num,
        use_lora=False,
        use_low_rank_linear=False,
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
    print(f"`MedusaModel` has {total_params:,} parameters.")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"`MedusaModel` has {trainable_params:,} trainable parameters.")

    # Load dataset
    # dataset = load_from_disk()
    # dataset = load_dataset("shibing624/sharegpt_gpt4")
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
    
    samples = dataset["train"]["conversations"]
    new_samples = []
    for sample in tqdm(samples):
        new_sample = [{"role": items["from"].replace("human", "user").replace("gpt", "assistant"), "content": items["value"]} for items in sample]
        new_samples.append(new_sample)

    train_samples, eval_samples = train_test_split(new_samples, test_size=0.1, random_state=2999)
    # train_samples, eval_samples = train_test_split(samples, test_size=0.1, random_state=2999)
    print("Data Size:", len(samples))

    # Tokenized
    train_samples = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in train_samples if messages[0]["role"] != "assistant"]
    eval_samples = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in eval_samples if messages[0]["role"] != "assistant"]

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
                num_logits_to_keep=max_length+head_num+1,
            )

            logits = outputs.logits

            raw_logits = logits[:1, ...]
            heads_logits = logits[1:, :, :-head_num-1, :]

            # Reshape and offset the raw_logits, and the shape must to satisfy the heads_logits
            target_labels, target_attention_mask = reshape_labels_with_offset(
                input_ids=input_ids,
                head_num=head_num,
                tokenizer=tokenizer,
                raw_attention_mask=attention_mask,
            )

            # Compute the log probabilities for both models
            # draft_log_probs = torch.nn.functional.log_softmax(heads_logits, dim=-1)

            loss = 0.0
            top_k_accept = {
                1: {
                    "total": [],
                    "accept": [],
                },
                3: {
                    "total": [],
                    "accept": [],
                },
            }

            for k in range(head_num):
                curr_lambda_k = pow(lambda_k, k + 1)

                # Extract the k-th head's logits
                k_head_logits = heads_logits[k].contiguous().view(-1, heads_logits.shape[-1])  # Shape: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
                k_target_labels = target_labels[k].contiguous().view(-1)

                # Compute hard labels and soft labels cross entropy loss
                hard_label_loss = torch.nn.functional.cross_entropy(
                    k_head_logits,  # Flatten logits
                    k_target_labels,  # Flatten hard labels
                    ignore_index=tokenizer.pad_token_id,
                )

                not_ignore = k_target_labels.ne(tokenizer.pad_token_id)

                for _k in [1, 3]:
                    _, topk = k_head_logits.topk(_k, dim=-1)

                    # Filtered out ignored positions (pad_token)
                    not_ignore_topk = topk[not_ignore]
                    not_ignore_k_target_labels = k_target_labels[not_ignore]

                    # Check if predictions match the target labels
                    correct = not_ignore_topk.eq(not_ignore_k_target_labels.unsqueeze(-1)).any(-1)

                    top_k_accept[_k]["total"].append(not_ignore.sum().item())
                    top_k_accept[_k]["accept"].append(correct.sum().item())

                head_loss = hard_label_loss
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

                top_1_accept = top_k_accept[1]["accept"][-1] / top_k_accept[1]["total"][-1]
                top_3_accept = top_k_accept[3]["accept"][-1] / top_k_accept[3]["total"][-1]
                print(f"Train - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(train_dataloader)}], Training Loss: {avg_loss:.4f}, Top-1 Acc: {top_1_accept}, Top-3 Acc: {top_3_accept}")

        # Evaluate the model
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader, 1):
                input_ids, attention_mask = batch

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_logits_to_keep=max_length+head_num+1,
                )

                logits = outputs.logits
                raw_logits = logits[:1, ...]
                heads_logits = logits[1:, :, :-head_num-1, :]

                # Reshape and offset the raw_logits, and the shape must to satisfy the heads_logits
                target_labels, target_attention_mask = reshape_labels_with_offset(
                    input_ids=input_ids,
                    head_num=head_num,
                    tokenizer=tokenizer,
                    raw_attention_mask=attention_mask,
                )

                loss = 0.0
                top_k_accept = {
                    1: {
                        "total": [],
                        "accept": [],
                    },
                    3: {
                        "total": [],
                        "accept": [],
                    },
                }

                for k in range(head_num):
                    curr_lambda_k = pow(lambda_k, k + 1)

                    # Extract the k-th head's logits
                    k_head_logits = heads_logits[k].contiguous().view(-1, heads_logits.shape[-1])  # Shape: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
                    k_target_labels = target_labels[k].contiguous().view(-1)

                    # Compute hard labels and soft labels cross entropy loss
                    hard_label_loss = torch.nn.functional.cross_entropy(
                        k_head_logits,  # Flatten logits
                        k_target_labels,  # Flatten hard labels
                        ignore_index=tokenizer.pad_token_id,
                    )

                    not_ignore = k_target_labels.ne(tokenizer.pad_token_id)

                    for _k in [1, 3]:
                        _, topk = k_head_logits.topk(_k, dim=-1)

                        # Filtered out ignored positions (pad_token)
                        not_ignore_topk = topk[not_ignore]
                        not_ignore_k_target_labels = k_target_labels[not_ignore]

                        # Check if predictions match the target labels
                        correct = not_ignore_topk.eq(not_ignore_k_target_labels.unsqueeze(-1)).any(-1)

                        top_k_accept[_k]["total"].append(not_ignore.sum().item())
                        top_k_accept[_k]["accept"].append(correct.sum().item())

                    head_loss = hard_label_loss
                    loss += curr_lambda_k * head_loss

                eval_loss += loss.item()
                eval_loss_history.append(loss.item())

                avg_loss = eval_loss / batch_idx

                top_1_accept = top_k_accept[1]["accept"][-1] / top_k_accept[1]["total"][-1]
                top_3_accept = top_k_accept[3]["accept"][-1] / top_k_accept[3]["total"][-1]
                print(f"Eval - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(eval_dataloader)}], Eval Loss: {avg_loss:.4f}, Top-1 Acc: {top_1_accept}, Top-3 Acc: {top_3_accept}")

        # Save model checkpoint
        save_dir = "./checkpoints/vicuna_1.3_checkpoints_hce_linear_20241219/"
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
        model.save_heads(
            save_path,
            train_loss_history=train_loss_history,
            eval_loss_history=eval_loss_history,
        )
        print(f"checkpoint saved at {save_path}")
    

if __name__ == "__main__":
    main()
