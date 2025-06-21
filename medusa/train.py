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
from utils.utils import log_gpu_status_decorator


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
    raw_logits_labels: torch.LongTensor,
    input_ids: torch.LongTensor,
    head_num: int,
    tokenizer: AutoTokenizer,
    raw_attention_mask: torch.LongTensor = None,
) -> Tuple[torch.FloatTensor, Optional[torch.LongTensor]]:
    # input_ids shape: [batch_size, seq_len_with_head_num_padding]
    batch_size, seq_len_with_head_num_padding = input_ids.shape
    seq_len = seq_len_with_head_num_padding - head_num - 1

    # Init null logits
    updated_logits_labels = torch.full(
        (head_num, batch_size, seq_len),
        tokenizer.pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

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
        # Update logits labels
        start_idx = head_idx + 1
        end_idx = seq_len + head_idx + 1
        updated_logits_labels[head_idx] = raw_logits_labels[..., start_idx:end_idx]
        
        # Update labels
        start_idx = head_idx + 2
        end_idx = seq_len + head_idx + 2
        updated_labels[head_idx] = input_ids[:, start_idx:end_idx]

        if raw_attention_mask is not None:
            updated_attention_mask[head_idx] = raw_attention_mask[:, start_idx:end_idx]

    return updated_logits_labels, updated_labels, updated_attention_mask
    

@log_gpu_status_decorator(log_file_prefix="medusa_training", interval=5, gpu_ids=[0])
def main() -> None:
    # Settings
    epochs = 100
    batch_size = 1
    accumulation_steps = 4
    max_length = 512
    lr = 5e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha = 0.2
    head_num = 5
    lambda_k = 0.8
    use_self_generated_labels = True
    print(f"Device: {device}")

    # Load model and tokenizer
    # pretrained_model_name_or_path = "../models/meta-llama--Meta-Llama-3.1-8B-Instruct"
    pretrained_model_name_or_path = "../models/HuggingFaceTB--SmolLM2-135M-Instruct"
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get the maximum width of parameter count
    param_width = len(f"{total_params:,}")

    # Format and print
    print(f"`MedusaModel` has {total_params:,} parameters.")
    print(f"`MedusaModel` has {f'{trainable_params:,}'.rjust(param_width)} trainable parameters.")

    # Load dataset
    # dataset = load_from_disk()
    dataset = load_dataset("shibing624/sharegpt_gpt4")
    # dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
    
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
    diff_head_top_k_accept = [
        {
            1: {
                "total": [],
                "accept": [],
            },
            3: {
                "total": [],
                "accept": [],
            },
        } for _ in range(head_num)
    ]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loss_history = []
        heads_loss_histories = {k: [] for k in range(head_num)}
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

            # Reshape and offset the raw_logits, and the shape must to satisfy the heads_logits
            raw_logits = logits[:1, ...]
            raw_logits_labels = torch.argmax(raw_logits, dim=-1)
            heads_logits = logits[1:, :, :-head_num-1, :]

            target_logits_labels, target_labels, target_attention_mask = reshape_labels_with_offset(
                raw_logits_labels=raw_logits_labels,
                input_ids=input_ids,
                head_num=head_num,
                tokenizer=tokenizer,
                raw_attention_mask=attention_mask,
            )

            # Compute the log probabilities for both models
            # draft_log_probs = torch.nn.functional.log_softmax(heads_logits, dim=-1)

            loss = 0.0
            for k in range(head_num):
                curr_lambda_k = pow(lambda_k, k + 1)

                # Extract the k-th head's logits
                k_head_logits = heads_logits[k].contiguous().view(-1, heads_logits.shape[-1])  # Shape: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
                k_logits_labels = target_logits_labels[k].contiguous().view(-1)
                k_target_labels = target_labels[k].contiguous().view(-1)

                # Compute hard labels and soft labels cross entropy loss
                hard_label_loss = torch.nn.functional.cross_entropy(
                    k_head_logits,  # Flatten logits
                    k_logits_labels if use_self_generated_labels else k_target_labels,  # Flatten hard labels
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

                    diff_head_top_k_accept[k][_k]["total"].append(not_ignore.sum().item())
                    diff_head_top_k_accept[k][_k]["accept"].append(correct.sum().item())

                head_loss = hard_label_loss
                heads_loss_histories[k].append(head_loss.item())
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

                print("="*70)
                for k in range(head_num):
                    top_1_accept = diff_head_top_k_accept[k][1]["accept"][-1] / diff_head_top_k_accept[k][1]["total"][-1]
                    top_3_accept = diff_head_top_k_accept[k][3]["accept"][-1] / diff_head_top_k_accept[k][3]["total"][-1]
                    head_loss = sum(heads_loss_histories[k]) / len(heads_loss_histories[k])
                    print(f"Train - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(train_dataloader)}], Head [{k+1}/{head_num}], Training Loss: {head_loss:.4f}, Top-1 Acc: {top_1_accept:.4f}, Top-3 Acc: {top_3_accept:.4f}")

        # Evaluate the model
        model.eval()
        eval_loss = 0
        heads_loss_histories = {k: [] for k in range(head_num)}

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader, 1):
                input_ids, attention_mask = batch

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_logits_to_keep=max_length+head_num+1,
                )

                logits = outputs.logits

                # Reshape and offset the raw_logits, and the shape must to satisfy the heads_logits
                raw_logits = logits[:1, ...]
                raw_logits_labels = torch.argmax(raw_logits, dim=-1)
                heads_logits = logits[1:, :, :-head_num-1, :]

                target_logits_labels, target_labels, target_attention_mask = reshape_labels_with_offset(
                    raw_logits_labels=raw_logits_labels,
                    input_ids=input_ids,
                    head_num=head_num,
                    tokenizer=tokenizer,
                    raw_attention_mask=attention_mask,
                )
    
                loss = 0.0
                for k in range(head_num):
                    curr_lambda_k = pow(lambda_k, k + 1)

                    # Extract the k-th head's logits
                    k_head_logits = heads_logits[k].contiguous().view(-1, heads_logits.shape[-1])  # Shape: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
                    # k_head_logits = torch.nn.LogSoftmax(dim=-1)(k_head_logits)
                    k_logits_labels = target_logits_labels[k].contiguous().view(-1)
                    k_target_labels = target_labels[k].contiguous().view(-1)

                    # Compute hard labels and soft labels cross entropy loss
                    hard_label_loss = torch.nn.functional.cross_entropy(
                        k_head_logits,  # Flatten logits
                        k_logits_labels if use_self_generated_labels else k_target_labels,  # Flatten hard labels
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

                        diff_head_top_k_accept[k][_k]["total"].append(not_ignore.sum().item())
                        diff_head_top_k_accept[k][_k]["accept"].append(correct.sum().item())

                    head_loss = hard_label_loss
                    heads_loss_histories[k].append(head_loss.item())
                    loss += curr_lambda_k * head_loss

                eval_loss += loss.item()
                eval_loss_history.append(loss.item())

                avg_loss = eval_loss / batch_idx

                print("="*70)
                for k in range(head_num):
                    top_1_accept = diff_head_top_k_accept[k][1]["accept"][-1] / diff_head_top_k_accept[k][1]["total"][-1]
                    top_3_accept = diff_head_top_k_accept[k][3]["accept"][-1] / diff_head_top_k_accept[k][3]["total"][-1]
                    head_loss = sum(heads_loss_histories[k]) / len(heads_loss_histories[k])
                    print(f"Eval - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(train_dataloader)}], Head [{k+1}/{head_num}], Eval Loss: {head_loss:.4f}, Top-1 Acc: {top_1_accept:.4f}, Top-3 Acc: {top_3_accept:.4f}")

        # Save model checkpoint
        save_dir = "./checkpoints/llama_3_checkpoints_hce_linear_20250102/"
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
        model.save_heads(
            save_path,
            train_loss_history=train_loss_history,
            eval_loss_history=eval_loss_history,
        )
        print(f"checkpoint saved at {save_path}")
    

if __name__ == "__main__":
    main()
