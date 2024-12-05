from typing import Dict

import argparse
import os

from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from kangaroo_modeling.modeling_kangaroo_llama3 import KangarooLlamaForCausalLM


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs: Dict[str, torch.LongTensor], device: torch.DeviceObjType):
        self.inputs = inputs
        self.device = device

    def __len__(self) -> int:
        return self.inputs.input_ids.shape[0]

    def __getitem__(self, index: int):
        return (
            self.inputs.input_ids[index].to(self.device),
            self.inputs.attention_mask[index].to(self.device),
        )


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
    model = KangarooLlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.set_skip_layer(shallow_layer_num=args.shallow_layer_num)
    model.set_adapter_layer("decoder_layer")
    if args.adapter_dir:
        model.load_adapter(args.adapter_dir)

    model = model.to(device).eval()

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    samples = dataset["train"]["conversations"]
    samples = [[{"role": sample[0]["from"].replace("human", "user").replace("gpt", "assistant"), "content": sample[0]["value"]}] for sample in samples]

    # Tokenized
    print("Tokenizing...")
    inputs = tokenizer(
        [tokenizer.apply_chat_template(messages, tokenize=False) for messages in samples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    # Dataloader
    custom_dataset = CustomDataset(inputs=inputs, device=device)
    dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=args.prepare_batch_size, shuffle=False)

    # Preparing
    for batch_idx, batch in enumerate(tqdm(dataloader), 1):
        input_ids, attention_mask = batch

        with torch.no_grad():
            shallow_hidden_states, target_hidden_states = model.prepare_dataset(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_logits_to_keep=args.max_length,
            )

            shallow_hidden_states = shallow_hidden_states.clone().detach().cpu()
            target_hidden_states = target_hidden_states.clone().detach().cpu()

        # Save
        torch.save(shallow_hidden_states, os.path.join(args.output_dir, f"shallow_{batch_idx}.pt"))
        torch.save(target_hidden_states, os.path.join(args.output_dir, f"target_{batch_idx}.pt"))

    # Save lm_head
    model.save_head(save_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for KangarooLlama model.")

    # Add arguments
    parser.add_argument("--model_path", type=str, default="../models/meta-llama--Meta-Llama-3.1-8B-Instruct", help="Path to the pretrained model.")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to the adapter directory.")
    parser.add_argument("--dataset_name", type=str, default="shibing624/sharegpt_gpt4", help="HuggingFace dataset name.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--prepare_batch_size", type=int, default=1, help="Batch size for data preparation.")
    parser.add_argument("--shallow_layer_num", type=int, default=2, help="Number of shallow layers to skip.")
    parser.add_argument("--output_dir", type=str, default="/mnt/kangaroo_train_data/llama-3.1-8b_train_data", help="File to save the prepared training data.")

    # Parse arguments
    args = parser.parse_args()

    # Run data preparation
    get_all_train_data(args)
