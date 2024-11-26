from typing import Dict, List

from datasets import load_dataset

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

from kangaroo_modeling.modeling_kangaroo_llama3 import KangarooLlamaForCausalLM


class CustomDataset(Dataset):
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
    

def main() -> None:
    # Settings
    epochs = 5
    batch_size = 4
    lr = 1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    pretrained_model_name_or_path = "../models/HuggingFaceTB--SmolLM2-135M-Instruct"
    model = KangarooLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    model.set_skip_layer(shallow_layer_num=10)
    model.set_train_mode()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze adapter layer
    model.draft_mode_adapter_layer.train()
    for param in model.draft_mode_adapter_layer.parameters():
        param.requires_grad = True

    # Load dataset
    dataset = load_dataset("shibing624/sharegpt_gpt4")
    
    samples = dataset["train"]["conversations"]
    samples = [[{"role": sample[0]["from"].replace("human", "user").replace("gpt", "assistant"), "content": sample[0]["value"]}] for sample in samples]

    train_samples, eval_samples = train_test_split(samples, test_size=0.1, random_state=2999)

    # Tokenized
    train_inputs = tokenizer(
        [tokenizer.apply_chat_template(messages, tokenize=False) for messages in train_samples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)    

    eval_inputs = tokenizer(
        [tokenizer.apply_chat_template(messages, tokenize=False) for messages in eval_samples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)    

    train_dataset = CustomDataset(inputs=train_inputs, device=device)
    eval_dataset = CustomDataset(inputs=eval_inputs, device=device)

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.draft_mode_adapter_layer.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader, 1):
            input_ids, attention_mask = batch

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Zero gradients
            optimizer.zero_grad()

            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()

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
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader, 1):
                input_ids, attention_mask = batch

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                eval_loss += outputs.loss.item()

                avg_loss = eval_loss / batch_idx
                print(f"Eval - Epoch [{epoch + 1}/{epochs}] Steps [{batch_idx}/{len(eval_dataloader)}], Eval Loss: {avg_loss:.4f}")

        # Save model checkpoint
        model.save_adapter(f"./checkpoints/epoch_{epoch+1}")
        print(f"Adapter checkpoint saved at ./checkpoints/epoch_{epoch+1}/")


if __name__ == "__main__":
    main()
