from typing import Dict, List, Tuple

import argparse
import os
from functools import partial

from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from medusa_modeling.modeling_medusa_llama import MedusaLlamaForCausalLM


class CustomMedusaDataset(torch.utils.data.Dataset):
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
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    tokenized_batch = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
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


def get_all_train_data(args) -> None:
    # Base case: confirm the `output_dir` is existed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = MedusaLlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    samples = dataset["train"]["conversations"]
    samples = [[{"role": sample[0]["from"].replace("human", "user").replace("gpt", "assistant"), "content": sample[0]["value"]}] for sample in samples]
    print("Data Size:", len(samples))

    # Dataset
    samples = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in samples]
    dataset = CustomMedusaDataset(input_data=samples)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.prepare_batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device, head_num=5),
    )

    # Preparing
    for batch_idx, batch in enumerate(tqdm(dataloader), 1):
        input_ids, attention_mask = batch

        with torch.no_grad():
            outputs = model.get_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            hidden_states = outputs[0]
            target_hidden_states = hidden_states.clone().detach().cpu()

        # Save
        torch.save(target_hidden_states, os.path.join(args.output_dir, f"target_{batch_idx}.pt"))
        print(f"data id {batch_idx} is saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for Medusa model.")

    # Add arguments
    parser.add_argument("--model_path", type=str, default="../models/meta-llama--Meta-Llama-3.1-8B-Instruct", help="Path to the pretrained model.")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to the adapter directory.")
    parser.add_argument("--dataset_name", type=str, default="shibing624/sharegpt_gpt4", help="HuggingFace dataset name.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--prepare_batch_size", type=int, default=1, help="Batch size for data preparation.")
    parser.add_argument("--shallow_layer_num", type=int, default=2, help="Number of shallow layers to skip.")
    parser.add_argument("--output_dir", type=str, default="/mnt/cached_train_data/llama-3.1-8b_train_data", help="File to save the prepared training data.")

    # Parse arguments
    args = parser.parse_args()

    # Run data preparation
    get_all_train_data(args)
